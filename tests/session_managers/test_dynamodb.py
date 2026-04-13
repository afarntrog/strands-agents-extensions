"""Tests for DynamoDBSessionManager."""

from decimal import Decimal

import boto3
import pytest
from moto import mock_aws
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage

from strands_agents_extensions.session_managers.dynamodb import (
    DynamoDBSessionManager,
    _convert_decimals_to_native_types,
)

TABLE_NAME = "test-sessions"
S3_BUCKET = "test-bucket"
S3_PREFIX = "test-prefix"
REGION = "us-east-1"


def _create_table(region=REGION):
    """Create a DynamoDB table for testing."""
    client = boto3.client("dynamodb", region_name=region)
    client.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    """Mock AWS credentials for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture()
def manager():
    """Create DynamoDBSessionManager with mocked AWS."""
    with mock_aws():
        _create_table()
        mgr = DynamoDBSessionManager(
            session_id="manager-session",
            table_name=TABLE_NAME,
            region_name=REGION,
        )
        yield mgr


@pytest.fixture()
def manager_with_s3():
    """Create DynamoDBSessionManager with S3 offloading."""
    with mock_aws():
        _create_table()
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=S3_BUCKET)

        mgr = DynamoDBSessionManager(
            session_id="manager-session",
            table_name=TABLE_NAME,
            region_name=REGION,
            s3_bucket=S3_BUCKET,
            s3_prefix=S3_PREFIX,
        )
        yield mgr


@pytest.fixture()
def session():
    """Create test session."""
    from strands.types.session import SessionType

    return Session(session_id="test-session", session_type=SessionType.AGENT)


@pytest.fixture()
def agent():
    """Create test agent."""
    return SessionAgent(agent_id="test-agent", state={}, conversation_manager_state={})


@pytest.fixture()
def message():
    """Create test message."""
    from strands.types.content import Message

    return SessionMessage(message_id=1, message=Message(role="user", content=[{"text": "Hello"}]))


# ─── Initialization ────────────────────────────────────────────


class TestInitialization:
    """Test DynamoDBSessionManager initialization."""

    def test_init_with_defaults(self):
        with mock_aws():
            _create_table()
            mgr = DynamoDBSessionManager(
                session_id="test",
                table_name=TABLE_NAME,
                region_name=REGION,
            )
            assert mgr._table_name == TABLE_NAME
            assert mgr._table is not None
            assert mgr._s3_client is None

    def test_init_with_s3_bucket(self):
        with mock_aws():
            _create_table()
            boto3.client("s3", region_name=REGION).create_bucket(Bucket=S3_BUCKET)
            mgr = DynamoDBSessionManager(
                session_id="test",
                table_name=TABLE_NAME,
                region_name=REGION,
                s3_bucket=S3_BUCKET,
                s3_prefix=S3_PREFIX,
            )
            assert mgr._s3_client is not None
            assert mgr._s3_bucket == S3_BUCKET
            assert mgr._s3_prefix == S3_PREFIX

    def test_init_without_s3_bucket(self):
        with mock_aws():
            _create_table()
            mgr = DynamoDBSessionManager(
                session_id="test",
                table_name=TABLE_NAME,
                region_name=REGION,
            )
            assert mgr._s3_client is None
            assert mgr._s3_bucket is None

    def test_init_with_custom_max_item_size(self):
        with mock_aws():
            _create_table()
            mgr = DynamoDBSessionManager(
                session_id="test",
                table_name=TABLE_NAME,
                region_name=REGION,
                max_item_size=200_000,
            )
            assert mgr._max_item_size == 200_000


# ─── Session CRUD ──────────────────────────────────────────────


class TestSessionOperations:
    """Test session CRUD operations."""

    def test_create_session(self, manager, session):
        manager.create_session(session)
        retrieved = manager.read_session(session.session_id)
        assert retrieved.session_id == session.session_id

    def test_create_duplicate_session(self, manager, session):
        manager.create_session(session)
        with pytest.raises(SessionException, match="already exists"):
            manager.create_session(session)

    def test_read_nonexistent_session(self, manager):
        result = manager.read_session("nonexistent")
        assert result is None

    def test_delete_session(self, manager, session):
        manager.create_session(session)
        manager.delete_session(session.session_id)
        result = manager.read_session(session.session_id)
        assert result is None

    def test_delete_nonexistent_session(self, manager):
        with pytest.raises(SessionException, match="not found"):
            manager.delete_session("nonexistent")

    def test_delete_session_cascades(self, manager, session, agent, message):
        """Deleting a session removes all agents, messages, and multi-agent items."""
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        manager.create_message(session.session_id, agent.agent_id, message)
        manager.create_multi_agent(session.session_id, "multi-1", {"key": "value"})

        manager.delete_session(session.session_id)

        assert manager.read_session(session.session_id) is None
        assert manager.read_agent(session.session_id, agent.agent_id) is None
        assert manager.list_messages(session.session_id, agent.agent_id) == []
        assert manager.read_multi_agent(session.session_id, "multi-1") is None


# ─── Agent CRUD ────────────────────────────────────────────────


class TestAgentOperations:
    """Test agent CRUD operations."""

    def test_create_agent(self, manager, session, agent):
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        retrieved = manager.read_agent(session.session_id, agent.agent_id)
        assert retrieved.agent_id == agent.agent_id
        assert retrieved.state == agent.state

    def test_read_nonexistent_agent(self, manager, session):
        manager.create_session(session)
        result = manager.read_agent(session.session_id, "nonexistent")
        assert result is None

    def test_update_agent(self, manager, session, agent):
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        agent.state = {"updated": "value"}
        manager.update_agent(session.session_id, agent)

        retrieved = manager.read_agent(session.session_id, agent.agent_id)
        assert retrieved.state == {"updated": "value"}

    def test_update_nonexistent_agent(self, manager, session, agent):
        manager.create_session(session)
        with pytest.raises(SessionException, match="not found"):
            manager.update_agent(session.session_id, agent)

    def test_read_agent_decimal_conversion(self, manager, session):
        """Agent state with numeric values should return int/float, not Decimal."""
        manager.create_session(session)
        agent = SessionAgent(
            agent_id="numeric-agent",
            state={"count": 42, "rate": 3.14},
            conversation_manager_state={},
        )
        manager.create_agent(session.session_id, agent)

        retrieved = manager.read_agent(session.session_id, "numeric-agent")
        assert isinstance(retrieved.state["count"], int)
        assert isinstance(retrieved.state["rate"], float)


# ─── Message CRUD + List ───────────────────────────────────────


class TestMessageOperations:
    """Test message CRUD operations."""

    def test_create_message(self, manager, session, agent, message):
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        manager.create_message(session.session_id, agent.agent_id, message)

        retrieved = manager.read_message(session.session_id, agent.agent_id, message.message_id)
        assert retrieved.message_id == message.message_id

    def test_read_nonexistent_message(self, manager, session, agent):
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        result = manager.read_message(session.session_id, agent.agent_id, 999)
        assert result is None

    def test_update_message(self, manager, session, agent, message):
        from strands.types.content import Message

        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        manager.create_message(session.session_id, agent.agent_id, message)

        message.message = Message(role="user", content=[{"text": "Updated"}])
        manager.update_message(session.session_id, agent.agent_id, message)

        retrieved = manager.read_message(session.session_id, agent.agent_id, message.message_id)
        assert retrieved.message["content"][0]["text"] == "Updated"

    def test_update_nonexistent_message(self, manager, session, agent, message):
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)
        with pytest.raises(SessionException, match="not found"):
            manager.update_message(session.session_id, agent.agent_id, message)

    def test_list_messages_all(self, manager, session, agent):
        from strands.types.content import Message

        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        for i in range(5):
            msg = SessionMessage(message_id=i, message=Message(role="user", content=[{"text": f"Message {i}"}]))
            manager.create_message(session.session_id, agent.agent_id, msg)

        messages = manager.list_messages(session.session_id, agent.agent_id)
        assert len(messages) == 5
        assert messages[0].message_id == 0
        assert messages[4].message_id == 4

    def test_list_messages_with_limit(self, manager, session, agent):
        from strands.types.content import Message

        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        for i in range(10):
            msg = SessionMessage(message_id=i, message=Message(role="user", content=[{"text": f"Message {i}"}]))
            manager.create_message(session.session_id, agent.agent_id, msg)

        messages = manager.list_messages(session.session_id, agent.agent_id, limit=3)
        assert len(messages) == 3
        assert messages[0].message_id == 0

    def test_list_messages_with_offset(self, manager, session, agent):
        from strands.types.content import Message

        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        for i in range(10):
            msg = SessionMessage(message_id=i, message=Message(role="user", content=[{"text": f"Message {i}"}]))
            manager.create_message(session.session_id, agent.agent_id, msg)

        messages = manager.list_messages(session.session_id, agent.agent_id, offset=5)
        assert len(messages) == 5
        assert messages[0].message_id == 5

    def test_list_messages_with_limit_and_offset(self, manager, session, agent):
        from strands.types.content import Message

        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        for i in range(10):
            msg = SessionMessage(message_id=i, message=Message(role="user", content=[{"text": f"Message {i}"}]))
            manager.create_message(session.session_id, agent.agent_id, msg)

        messages = manager.list_messages(session.session_id, agent.agent_id, limit=3, offset=2)
        assert len(messages) == 3
        assert messages[0].message_id == 2
        assert messages[2].message_id == 4


# ─── Multi-Agent CRUD ──────────────────────────────────────────


class TestMultiAgentOperations:
    """Test multi-agent state operations."""

    def test_create_multi_agent(self, manager, session):
        manager.create_session(session)
        state = {"agents": ["agent1", "agent2"], "current": "agent1"}
        manager.create_multi_agent(session.session_id, "multi-1", state)

        retrieved = manager.read_multi_agent(session.session_id, "multi-1")
        assert retrieved == state

    def test_read_nonexistent_multi_agent(self, manager, session):
        manager.create_session(session)
        result = manager.read_multi_agent(session.session_id, "nonexistent")
        assert result is None

    def test_update_multi_agent(self, manager, session):
        manager.create_session(session)
        state = {"agents": ["agent1"], "current": "agent1"}
        manager.create_multi_agent(session.session_id, "multi-1", state)

        new_state = {"agents": ["agent1", "agent2"], "current": "agent2"}
        manager.update_multi_agent(session.session_id, "multi-1", new_state)

        retrieved = manager.read_multi_agent(session.session_id, "multi-1")
        assert retrieved == new_state

    def test_update_nonexistent_multi_agent(self, manager, session):
        manager.create_session(session)
        with pytest.raises(SessionException, match="not found"):
            manager.update_multi_agent(session.session_id, "nonexistent", {})


# ─── S3 Offloading ─────────────────────────────────────────────


class TestS3Offloading:
    """Test S3 offloading for large messages."""

    def _make_large_message(self, message_id=1, size_bytes=500_000):
        from strands.types.content import Message

        large_text = "x" * size_bytes
        return SessionMessage(
            message_id=message_id,
            message=Message(role="user", content=[{"text": large_text}]),
        )

    def test_create_large_message_s3_offload(self, manager_with_s3, session, agent):
        """Large message is offloaded to S3 with reference in DynamoDB."""
        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        retrieved = manager_with_s3.read_message(session.session_id, agent.agent_id, large_msg.message_id)
        assert retrieved.message_id == large_msg.message_id
        assert retrieved.message["content"][0]["text"] == "x" * 500_000

    def test_read_large_message_from_s3(self, manager_with_s3, session, agent):
        """Large message is transparently loaded from S3 on read."""
        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        retrieved = manager_with_s3.read_message(session.session_id, agent.agent_id, 1)
        assert "x" * 500_000 in retrieved.message["content"][0]["text"]

    def test_list_large_messages_from_s3(self, manager_with_s3, session, agent):
        """list_messages returns both normal and S3-offloaded messages."""
        from strands.types.content import Message

        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        small_msg = SessionMessage(message_id=0, message=Message(role="user", content=[{"text": "small"}]))
        manager_with_s3.create_message(session.session_id, agent.agent_id, small_msg)

        large_msg = self._make_large_message(message_id=1)
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        messages = manager_with_s3.list_messages(session.session_id, agent.agent_id)
        assert len(messages) == 2
        assert messages[0].message["content"][0]["text"] == "small"
        assert len(messages[1].message["content"][0]["text"]) == 500_000

    def test_update_large_message_s3(self, manager_with_s3, session, agent):
        """Updating a large message updates S3 content."""
        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        new_large_msg = self._make_large_message(size_bytes=600_000)
        manager_with_s3.update_message(session.session_id, agent.agent_id, new_large_msg)

        retrieved = manager_with_s3.read_message(session.session_id, agent.agent_id, 1)
        assert len(retrieved.message["content"][0]["text"]) == 600_000

    def test_s3_without_bucket_configured(self, manager, session, agent):
        """Large message fails gracefully when no S3 bucket is configured and item exceeds DynamoDB limit."""
        manager.create_session(session)
        manager.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        with pytest.raises(SessionException, match="Failed to create message"):
            manager.create_message(session.session_id, agent.agent_id, large_msg)

    def test_s3_prefix(self, manager_with_s3, session, agent):
        """S3 key includes the configured prefix."""
        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        s3 = manager_with_s3._s3_client
        objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        assert objects["KeyCount"] > 0
        key = objects["Contents"][0]["Key"]
        assert key.startswith(S3_PREFIX)

    def test_delete_session_with_s3_cleanup(self, manager_with_s3, session, agent):
        """Deleting a session also cleans up S3 objects."""
        manager_with_s3.create_session(session)
        manager_with_s3.create_agent(session.session_id, agent)

        large_msg = self._make_large_message()
        manager_with_s3.create_message(session.session_id, agent.agent_id, large_msg)

        s3 = manager_with_s3._s3_client
        objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        assert objects["KeyCount"] > 0

        manager_with_s3.delete_session(session.session_id)

        objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        assert objects.get("KeyCount", 0) == 0


# ─── ID Validation ─────────────────────────────────────────────


class TestIDValidation:
    """Test ID validation."""

    def test_id_with_hash_rejected(self, manager):
        """IDs containing '#' raise ValueError."""
        from strands.types.session import SessionType

        session = Session(session_id="bad#id", session_type=SessionType.AGENT)
        with pytest.raises(ValueError, match="cannot contain '#'"):
            manager.create_session(session)

    def test_id_with_underscore_allowed(self, manager):
        """IDs with underscores work fine."""
        from strands.types.session import SessionType

        session = Session(session_id="my_custom_id", session_type=SessionType.AGENT)
        manager.create_session(session)
        retrieved = manager.read_session("my_custom_id")
        assert retrieved.session_id == "my_custom_id"

    def test_id_with_dash_allowed(self, manager):
        """Standard UUID-style IDs work."""
        from strands.types.session import SessionType

        session = Session(session_id="abc-123-def", session_type=SessionType.AGENT)
        manager.create_session(session)
        retrieved = manager.read_session("abc-123-def")
        assert retrieved.session_id == "abc-123-def"

    def test_agent_id_with_hash_rejected(self, manager, session):
        """Agent IDs containing '#' raise ValueError."""
        manager.create_session(session)
        bad_agent = SessionAgent(agent_id="bad#agent", state={}, conversation_manager_state={})
        with pytest.raises(ValueError, match="cannot contain '#'"):
            manager.create_agent(session.session_id, bad_agent)


# ─── Utility ───────────────────────────────────────────────────


class TestUtility:
    """Test utility functions."""

    def test_convert_decimals_to_native_types_int(self):
        result = _convert_decimals_to_native_types({"count": Decimal("42")})
        assert result == {"count": 42}
        assert isinstance(result["count"], int)

    def test_convert_decimals_to_native_types_float(self):
        result = _convert_decimals_to_native_types({"rate": Decimal("3.14")})
        assert result == {"rate": 3.14}
        assert isinstance(result["rate"], float)

    def test_convert_decimals_nested(self):
        data = {
            "outer": {
                "inner": Decimal("10"),
                "list": [Decimal("1"), Decimal("2.5")],
            }
        }
        result = _convert_decimals_to_native_types(data)
        assert result == {"outer": {"inner": 10, "list": [1, 2.5]}}
        assert isinstance(result["outer"]["inner"], int)
        assert isinstance(result["outer"]["list"][1], float)

    def test_convert_decimals_preserves_non_decimal(self):
        data = {"name": "test", "flag": True, "items": [1, "two"]}
        result = _convert_decimals_to_native_types(data)
        assert result == data
