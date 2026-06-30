"""Test command registry — registration, lookup, system prompt."""

from __future__ import annotations

from commands.registry import Command, CommandRegistry, TriggerMode


class TestCommandRegistry:
    def test_register_and_get(self):
        reg = CommandRegistry()
        cmd = Command(name="test", description="A test command", trigger_modes=[TriggerMode.USER_TRIGGER], handler=lambda _: "mock")
        reg.register(cmd)
        assert reg.get("test") is cmd

    def test_get_unknown_returns_none(self):
        reg = CommandRegistry()
        assert reg.get("nonexistent") is None

    def test_get_case_sensitive(self):
        reg = CommandRegistry()
        cmd = Command(name="TestCmd", description="case test", trigger_modes=[TriggerMode.USER_TRIGGER], handler=lambda _: "mock")
        reg.register(cmd)
        assert reg.get("TestCmd") is cmd
        assert reg.get("testcmd") is None

    def test_build_system_prompt_addon_with_commands(self):
        reg = CommandRegistry()
        reg.register(Command(
            name="verify",
            description="验证模型适配",
            trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND],
            handler=lambda _: "mock",
            example="/verify 验证 Qwen3-8B",
        ))
        reg.register(Command(
            name="search",
            description="搜索模型",
            trigger_modes=[TriggerMode.LLM_AUTO, TriggerMode.LLM_RECOMMEND],
            handler=lambda _: "mock",
        ))
        prompt = reg.build_system_prompt_addon()
        assert "/verify" in prompt
        assert "/search" in prompt
        assert "Qwen3-8B" in prompt

    def test_build_system_prompt_addon_empty(self):
        reg = CommandRegistry()
        prompt = reg.build_system_prompt_addon()
        assert prompt == ""

    def test_multiple_registrations(self):
        reg = CommandRegistry()
        for i in range(5):
            reg.register(Command(name=f"cmd{i}", description=f"Command {i}", trigger_modes=[TriggerMode.USER_TRIGGER], handler=lambda _: "mock"))
        for i in range(5):
            assert reg.get(f"cmd{i}") is not None

    def test_trigger_mode_membership(self):
        cmd = Command(name="x", description="x", trigger_modes=[TriggerMode.USER_TRIGGER, TriggerMode.LLM_RECOMMEND], handler=lambda _: "mock")
        assert TriggerMode.USER_TRIGGER in cmd.trigger_modes
        assert TriggerMode.LLM_RECOMMEND in cmd.trigger_modes
        assert TriggerMode.LLM_AUTO not in cmd.trigger_modes
