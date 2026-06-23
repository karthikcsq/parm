# Active And Passive Memory Recall

Memory retrieval has two distinct modes.

**Active recall** happens when the user or agent deliberately searches for a
missing piece of information needed to complete the task. The information need
is already explicit: the system knows what it is looking for, so retrieval is
initiated by the query, plan, or tool call.

**Passive recall** happens when new information enters the context and related
memory surfaces because it connects to that information, even though searching
for related memories was not the original intent of the task. The relevance is
recognized only after the new entity, fact, or situation appears.

PARM targets passive recall. It watches the model's output and tool responses
for newly introduced entities, then checks whether those entities connect to
latent personal memory in a way that advances the user's goal.
