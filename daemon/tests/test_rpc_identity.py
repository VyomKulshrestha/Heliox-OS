from pilot.security.rpc_identity import (
    RpcClientRole,
    authenticate_rpc_client,
    derive_neural_signing_key,
    rpc_method_allowed,
)


def test_rpc_tokens_create_distinct_roles_without_empty_fallback() -> None:
    assert (
        authenticate_rpc_client(
            "ui-secret",
            ui_token="ui-secret",
            neural_token="neural-secret",
            mcp_token="mcp-secret",
        )
        == RpcClientRole.UI
    )
    assert (
        authenticate_rpc_client(
            "neural-secret",
            ui_token="ui-secret",
            neural_token="neural-secret",
            mcp_token="mcp-secret",
        )
        == RpcClientRole.NEURAL_SIDECAR
    )
    assert (
        authenticate_rpc_client(
            "mcp-secret",
            ui_token="ui-secret",
            neural_token="neural-secret",
            mcp_token="mcp-secret",
        )
        == RpcClientRole.MCP_LOCAL
    )
    assert authenticate_rpc_client("", ui_token="", neural_token="", mcp_token="") is None
    assert authenticate_rpc_client(None, ui_token="ui", neural_token="neural") is None


def test_neural_sidecar_cannot_call_general_or_confirmation_methods() -> None:
    role = RpcClientRole.NEURAL_SIDECAR
    assert rpc_method_allowed(role, "neural_intent_preview") is True
    assert rpc_method_allowed(role, "execute") is False
    assert rpc_method_allowed(role, "confirm") is False
    assert rpc_method_allowed(role, "update_config") is False
    assert rpc_method_allowed(RpcClientRole.UI, "execute") is True


def test_local_mcp_can_only_call_its_bounded_adapter_methods() -> None:
    role = RpcClientRole.MCP_LOCAL
    assert rpc_method_allowed(role, "health") is True
    assert rpc_method_allowed(role, "mcp_plan_task") is True
    assert rpc_method_allowed(role, "mcp_typed_decision") is True
    assert rpc_method_allowed(role, "mcp_submit_task") is True
    assert rpc_method_allowed(role, "mcp_task_status") is True
    assert rpc_method_allowed(role, "mcp_cancel_task") is True
    assert rpc_method_allowed(role, "execute") is False
    assert rpc_method_allowed(role, "confirm") is False
    assert rpc_method_allowed(role, "update_config") is False
    assert rpc_method_allowed(role, "store_api_key") is False


def test_intent_signing_key_is_stable_domain_separated_and_bounded() -> None:
    token = "n" * 43
    assert derive_neural_signing_key(token) == derive_neural_signing_key(token)
    assert derive_neural_signing_key(token) != token.encode()
    assert len(derive_neural_signing_key(token)) == 32
