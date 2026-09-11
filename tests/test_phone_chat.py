import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from astra_harness.phone_chat import PhoneChat
from astra_harness.codex_runtime import RecoveryRequired, RuntimeFailure


class PhoneChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.settings = patch("astra_harness.phone_chat.openrouter_settings", return_value={
            "model": "z-ai/glm-5.3-flash", "api_key": "test-secret"})
        self.settings.start()
        self.which = patch("astra_harness.chat_cli.shutil.which", return_value="/native/codex")
        self.which.start()

    async def asyncTearDown(self):
        self.which.stop()
        self.settings.stop()
        self.tmp.cleanup()

    def process(self, answer="Hello", complete=True):
        reader = asyncio.StreamReader()
        events = [{"type": "thread.started", "thread_id": "test-thread"},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": answer}}]
        if complete:
            events.append({"type": "turn.completed"})
        for event in events:
            reader.feed_data((json.dumps(event) + "\n").encode())
        reader.feed_eof()
        from unittest.mock import Mock
        return Mock(stdout=reader, stdin=Mock(drain=AsyncMock()), returncode=0, wait=AsyncMock(return_value=0))

    async def test_persistent_resume_and_duplicate_without_generation(self):
        chat = PhoneChat(self.path)
        await chat.start()
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=self.process())) as spawn:
            self.assertEqual(await chat.reply("one", "hi"), "Hello")
            self.assertEqual(await chat.reply("one", "hi"), "Hello")
            self.assertEqual(spawn.await_count, 1)
            argv = spawn.call_args.args
            self.assertNotIn("test-secret", str(argv))
            self.assertEqual(argv[argv.index("--ask-for-approval") + 1], "never")
            self.assertEqual(argv[argv.index("--sandbox") + 1], "workspace-write")
        resumed = PhoneChat(self.path)
        await resumed.start()
        self.assertEqual(resumed.command()[-3:], ["resume", "test-thread", "-"])
        self.assertEqual(await resumed.reply("one", "hi"), "Hello")
        with self.assertRaises(RecoveryRequired):
            await resumed.reply("one", "changed")

    async def test_uncertain_turn_never_replayed(self):
        chat = PhoneChat(self.path)
        await chat.start()
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=self.process(complete=False))):
            with self.assertRaises(RuntimeFailure):
                await chat.reply("one", "hi")
        resumed = PhoneChat(self.path)
        await resumed.start()
        with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
            with self.assertRaises(RecoveryRequired):
                await resumed.reply("one", "hi")
            with self.assertRaises(RecoveryRequired):
                await resumed.reply("two", "another")
            spawn.assert_not_called()

    async def test_secret_echo_refused_and_long_replies_bounded(self):
        chat = PhoneChat(self.path)
        await chat.start()
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=self.process("x" * 5000))):
            self.assertLessEqual(len(await chat.reply("one", "hi")), 4000)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=self.process("test-secret"))):
            with self.assertRaises(RuntimeFailure):
                await chat.reply("two", "hi")
        self.assertNotIn("test-secret", chat.path.read_text())
