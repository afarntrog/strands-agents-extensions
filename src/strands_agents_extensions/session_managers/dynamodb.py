"""DynamoDB session manager implementation with optional S3 offloading for large messages."""

import json
import logging
import time
import uuid
from decimal import Decimal
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from strands.session import RepositorySessionManager, SessionRepository
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage

logger = logging.getLogger(__name__)

MAX_DYNAMODB_ITEM_SIZE = 400_000

# Marker key used in DynamoDB when content is offloaded to S3
_S3_REF_KEY = "__s3_ref"


def _convert_decimals_to_native_types(obj: Any) -> Any:
    """Recursively convert DynamoDB Decimal types to native int/float.

    DynamoDB returns all numbers as ``decimal.Decimal``. This walks an
    arbitrarily-nested structure and converts them to ``int`` (when there
    is no fractional part) or ``float``.
    """
    if isinstance(obj, list):
        return [_convert_decimals_to_native_types(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _convert_decimals_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    return obj


def _normalize_for_dynamodb(obj: Any) -> Any:
    """Normalize a Python dict for DynamoDB storage.

    Round-trips through JSON to convert enums, datetimes, and other
    non-primitive types into JSON-serializable equivalents, then converts
    floats to Decimal (required by the DynamoDB Resource API).
    """
    serializable = json.loads(json.dumps(obj, default=str))
    return _floats_to_decimal(serializable)


def _floats_to_decimal(obj: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [_floats_to_decimal(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    return obj


def _validate_id(value: str, label: str) -> None:
    """Reject IDs containing the ``#`` delimiter used in composite keys."""
    if "#" in value:
        raise ValueError(f"{label}={value} | id cannot contain '#' character")


class DynamoDBSessionManager(RepositorySessionManager, SessionRepository):
    """DynamoDB-based session manager with optional S3 offloading.

    Stores all session data in a single DynamoDB table using composite keys::

        PK: SESSION#{session_id}
        SK: SESSION | AGENT#{agent_id} | AGENT#{agent_id}#MSG#{message_id} | MULTIAGENT#{id}

    Messages exceeding ``max_item_size`` are automatically offloaded to S3
    and replaced with a reference pointer in DynamoDB.

    Args:
        session_id: Unique session identifier.
        table_name: DynamoDB table name.
        region_name: AWS region (uses default if ``None``).
        boto_session: Pre-configured boto3 session (uses default if ``None``).
        boto_client_config: Botocore client config (e.g. for retries).
        s3_bucket: S3 bucket for large-message offloading. ``None`` disables offloading.
        s3_prefix: Key prefix inside the S3 bucket.
        max_item_size: Byte threshold for S3 offloading (default 400KB).
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        session_id: str,
        table_name: str,
        region_name: str | None = None,
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str = "",
        max_item_size: int = MAX_DYNAMODB_ITEM_SIZE,
        **kwargs: Any,
    ) -> None:
        """Initialize DynamoDB session manager."""
        _validate_id(session_id, "session_id")

        self._table_name = table_name
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._max_item_size = max_item_size

        sess = boto_session or boto3.Session(region_name=region_name)
        cfg = boto_client_config or BotocoreConfig(
            user_agent_extra="strands-agents-extensions/dynamodb-session-manager",
        )

        self._ddb = sess.resource("dynamodb", config=cfg)
        self._table = self._ddb.Table(table_name)
        self._s3_client = sess.client("s3", config=cfg) if s3_bucket else None

        super().__init__(session_id=session_id, session_repository=self, **kwargs)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pk(session_id: str) -> str:
        return f"SESSION#{session_id}"

    @staticmethod
    def _session_sk() -> str:
        return "SESSION"

    @staticmethod
    def _agent_sk(agent_id: str) -> str:
        return f"AGENT#{agent_id}"

    @staticmethod
    def _message_sk(agent_id: str, message_id: int) -> str:
        return f"AGENT#{agent_id}#MSG#{message_id:010d}"

    @staticmethod
    def _message_sk_prefix(agent_id: str) -> str:
        return f"AGENT#{agent_id}#MSG#"

    @staticmethod
    def _multi_agent_sk(multi_agent_id: str) -> str:
        return f"MULTIAGENT#{multi_agent_id}"

    # ------------------------------------------------------------------
    # S3 offloading
    # ------------------------------------------------------------------

    def _s3_key(self, session_id: str, sort_key: str) -> str:
        prefix = f"{self._s3_prefix}/" if self._s3_prefix else ""
        return f"{prefix}{session_id}/{sort_key}/{uuid.uuid4()}.json"

    def _should_offload(self, data: dict[str, Any]) -> bool:
        """Check if data exceeds the max item size threshold."""
        return len(json.dumps(data).encode()) > self._max_item_size

    def _process_for_storage(self, data: dict[str, Any], session_id: str, sort_key: str) -> dict[str, Any]:
        """Offload data to S3 if it exceeds the threshold; return the item to store."""
        if not self._should_offload(data):
            return data

        if self._s3_client is None:
            logger.warning(
                "Message exceeds max_item_size (%d bytes) but no S3 bucket is configured. "
                "Storing in DynamoDB anyway — this may fail if the item exceeds DynamoDB's 400KB limit.",
                self._max_item_size,
            )
            return data

        key = self._s3_key(session_id, sort_key)
        self._s3_client.put_object(
            Bucket=self._s3_bucket,
            Key=key,
            Body=json.dumps(data).encode(),
        )
        return {_S3_REF_KEY: {"bucket": self._s3_bucket, "key": key}}

    def _resolve_s3_ref(self, data: dict[str, Any]) -> dict[str, Any]:
        """If data is an S3 reference, fetch the real payload."""
        if _S3_REF_KEY not in data:
            return data
        ref = data[_S3_REF_KEY]
        obj = self._s3_client.get_object(Bucket=ref["bucket"], Key=ref["key"])
        return json.loads(obj["Body"].read().decode())

    def _delete_s3_ref(self, data: dict[str, Any]) -> None:
        """Delete the S3 object backing an offloaded item, if present."""
        if self._s3_client and isinstance(data, dict) and _S3_REF_KEY in data:
            ref = data[_S3_REF_KEY]
            self._s3_client.delete_object(Bucket=ref["bucket"], Key=ref["key"])

    # ------------------------------------------------------------------
    # Paginated query helper
    # ------------------------------------------------------------------

    def _query_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Run a DynamoDB query with automatic pagination."""
        items: list[dict[str, Any]] = []
        response = self._table.query(**kwargs)
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = self._table.query(ExclusiveStartKey=response["LastEvaluatedKey"], **kwargs)
            items.extend(response.get("Items", []))
        return items

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session.

        Args:
            session: Session to create.
            **kwargs: Additional keyword arguments.

        Returns:
            The created session.

        Raises:
            SessionException: If the session already exists.
        """
        _validate_id(session.session_id, "session_id")
        pk = self._make_pk(session.session_id)
        sk = self._session_sk()
        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": _normalize_for_dynamodb(session.to_dict()),
        }
        try:
            self._table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self._ddb.meta.client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(f"Session {session.session_id} already exists") from exc
            raise SessionException(f"Failed to create session: {exc}") from exc
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Session | None:
        """Read a session by ID."""
        resp = self._table.get_item(
            Key={"pk": self._make_pk(session_id), "sk": self._session_sk()},
            ConsistentRead=True,
        )
        item = resp.get("Item")
        if not item:
            return None
        return Session.from_dict(_convert_decimals_to_native_types(item["data"]))

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete a session and all related items.

        Uses paginated query + batch delete. Also cleans up S3 objects.
        """
        pk = self._make_pk(session_id)

        items = self._query_all(
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": pk},
            ConsistentRead=True,
            ProjectionExpression="pk, sk, #d",
            ExpressionAttributeNames={"#d": "data"},
        )

        if not items:
            raise SessionException(f"Session {session_id} not found")

        # Clean up S3 references
        for item in items:
            data = item.get("data", {})
            if isinstance(data, dict):
                self._delete_s3_ref(data)

        # Batch delete in chunks of 25 (DynamoDB limit)
        for i in range(0, len(items), 25):
            batch = items[i : i + 25]
            request_items = {
                self._table_name: [{"DeleteRequest": {"Key": {"pk": it["pk"], "sk": it["sk"]}}} for it in batch]
            }
            response = self._ddb.meta.client.batch_write_item(RequestItems=request_items)

            # Retry unprocessed items
            unprocessed = response.get("UnprocessedItems", {})
            while unprocessed:
                response = self._ddb.meta.client.batch_write_item(RequestItems=unprocessed)
                unprocessed = response.get("UnprocessedItems", {})

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create an agent in a session."""
        _validate_id(session_agent.agent_id, "agent_id")

        if self.read_session(session_id) is None:
            raise SessionException(f"Session {session_id} not found")

        pk = self._make_pk(session_id)
        sk = self._agent_sk(session_agent.agent_id)
        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": _normalize_for_dynamodb(session_agent.to_dict()),
            "created_at": int(time.time()),
        }
        try:
            self._table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self._ddb.meta.client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(
                    f"Agent {session_agent.agent_id} already exists in session {session_id}"
                ) from exc
            raise SessionException(f"Failed to create agent: {exc}") from exc

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> SessionAgent | None:
        """Read an agent from a session."""
        resp = self._table.get_item(
            Key={"pk": self._make_pk(session_id), "sk": self._agent_sk(agent_id)},
            ConsistentRead=True,
        )
        item = resp.get("Item")
        if not item:
            return None
        return SessionAgent.from_dict(_convert_decimals_to_native_types(item["data"]))

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update an agent, preserving ``created_at``."""
        pk = self._make_pk(session_id)
        sk = self._agent_sk(session_agent.agent_id)

        resp = self._table.get_item(Key={"pk": pk, "sk": sk}, ConsistentRead=True)
        if "Item" not in resp:
            raise SessionException(f"Agent {session_agent.agent_id} not found in session {session_id}")

        created_at = resp["Item"].get("created_at", int(time.time()))
        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": _normalize_for_dynamodb(session_agent.to_dict()),
            "created_at": created_at,
            "updated_at": int(time.time()),
        }
        self._table.put_item(Item=item)

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a message for an agent, offloading to S3 if needed."""
        if self.read_agent(session_id, agent_id) is None:
            raise SessionException(f"Agent {agent_id} not found in session {session_id}")

        pk = self._make_pk(session_id)
        sk = self._message_sk(agent_id, session_message.message_id)
        data = self._process_for_storage(_normalize_for_dynamodb(session_message.to_dict()), session_id, sk)

        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": data,
        }
        try:
            self._table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self._ddb.meta.client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(f"Message {session_message.message_id} already exists") from exc
            raise SessionException(f"Failed to create message: {exc}") from exc

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> SessionMessage | None:
        """Read a single message, resolving S3 references."""
        resp = self._table.get_item(
            Key={"pk": self._make_pk(session_id), "sk": self._message_sk(agent_id, message_id)},
            ConsistentRead=True,
        )
        item = resp.get("Item")
        if not item:
            return None
        data = _convert_decimals_to_native_types(self._resolve_s3_ref(item["data"]))
        return SessionMessage.from_dict(data)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update a message, cleaning up old S3 objects if replaced."""
        pk = self._make_pk(session_id)
        sk = self._message_sk(agent_id, session_message.message_id)

        resp = self._table.get_item(Key={"pk": pk, "sk": sk}, ConsistentRead=True)
        if "Item" not in resp:
            raise SessionException(f"Message {session_message.message_id} not found")

        # Clean up old S3 object before writing new data
        old_data = resp["Item"].get("data", {})
        if isinstance(old_data, dict):
            self._delete_s3_ref(old_data)

        new_data = self._process_for_storage(_normalize_for_dynamodb(session_message.to_dict()), session_id, sk)
        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": new_data,
        }
        self._table.put_item(Item=item)

    def list_messages(
        self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        """List messages for an agent with pagination, resolving S3 references."""
        items = self._query_all(
            KeyConditionExpression="pk = :pk AND begins_with(sk, :sk_prefix)",
            ExpressionAttributeValues={
                ":pk": self._make_pk(session_id),
                ":sk_prefix": self._message_sk_prefix(agent_id),
            },
            ConsistentRead=True,
        )

        # Apply offset and limit in-memory (DynamoDB doesn't support offset natively)
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]

        messages = []
        for item in items:
            data = _convert_decimals_to_native_types(self._resolve_s3_ref(item["data"]))
            messages.append(SessionMessage.from_dict(data))
        return messages

    # ------------------------------------------------------------------
    # Multi-agent CRUD
    # ------------------------------------------------------------------

    def create_multi_agent(self, session_id: str, multi_agent_id: str, state: dict[str, Any], **kwargs: Any) -> None:
        """Create multi-agent state.

        Args:
            session_id: Session identifier.
            multi_agent_id: Multi-agent identifier.
            state: State dictionary to store.
            **kwargs: Additional keyword arguments.
        """
        _validate_id(multi_agent_id, "multi_agent_id")

        if self.read_session(session_id) is None:
            raise SessionException(f"Session {session_id} not found")

        pk = self._make_pk(session_id)
        sk = self._multi_agent_sk(multi_agent_id)
        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": _normalize_for_dynamodb(state),
        }
        try:
            self._table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(pk) AND attribute_not_exists(sk)",
            )
        except self._ddb.meta.client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(f"Multi-agent {multi_agent_id} already exists") from exc
            raise SessionException(f"Failed to create multi-agent: {exc}") from exc

    def read_multi_agent(self, session_id: str, multi_agent_id: str, **kwargs: Any) -> dict[str, Any] | None:
        """Read multi-agent state."""
        resp = self._table.get_item(
            Key={"pk": self._make_pk(session_id), "sk": self._multi_agent_sk(multi_agent_id)},
            ConsistentRead=True,
        )
        item = resp.get("Item")
        if not item:
            return None
        return _convert_decimals_to_native_types(item["data"])

    def update_multi_agent(self, session_id: str, multi_agent_id: str, state: dict[str, Any], **kwargs: Any) -> None:
        """Update multi-agent state.

        Args:
            session_id: Session identifier.
            multi_agent_id: Multi-agent identifier.
            state: Updated state dictionary.
            **kwargs: Additional keyword arguments.
        """
        pk = self._make_pk(session_id)
        sk = self._multi_agent_sk(multi_agent_id)

        resp = self._table.get_item(Key={"pk": pk, "sk": sk}, ConsistentRead=True)
        if "Item" not in resp:
            raise SessionException(f"Multi-agent {multi_agent_id} not found")

        item: dict[str, Any] = {
            "pk": pk,
            "sk": sk,
            "data": _normalize_for_dynamodb(state),
        }
        self._table.put_item(Item=item)
