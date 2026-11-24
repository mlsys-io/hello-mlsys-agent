"""The teacher agent: live Socratic coaching (`mlab ask`) and the final rubric assessment (`mlab assess`).

The teacher runs inside mlab, never inside the student's agent. It is any OpenAI-compatible chat model; the
reference is DeepSeek V4.1 Flash. Configure it with MLAB_TEACHER_BASE_URL, MLAB_TEACHER_MODEL and
MLAB_TEACHER_API_KEY. Under `./claude` the launcher holds those and serves a loopback relay instead, so the
student's agent only gets MLAB_TEACHER_URL and a session token, never the key.
"""
