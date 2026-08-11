import queue

# Global event queue for the agent.
# Background tasks and other asynchronous sources can put events here.
# The main agent loop can consume them to wake up early from sleeps.
EVENT_QUEUE = queue.Queue()
