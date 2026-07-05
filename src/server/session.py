"""A conversation session: the agent behind a transport-agnostic event stream.

handle(text) runs one turn and yields JSON events (see events.to_event), ending
with a {"type": "done"}. The agent is lazily constructed so importing this module
(and injecting a fake agent in tests) never spins up a real provider.
"""

from src.server.events import to_event


class AgentSession:
    def __init__(self, agent=None):
        self._agent = agent

    @property
    def agent(self):
        if self._agent is None:
            from agent import ArgentAgent
            self._agent = ArgentAgent()
        return self._agent

    def handle(self, text: str):
        """Run one turn, yielding normalized events then a terminal 'done'."""
        for chunk in self.agent.process_user_input(text):
            event = to_event(chunk)
            if event is not None:
                yield event
        yield {"type": "done"}
