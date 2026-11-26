"""Harold's rules: the Machine's constitution.

Harold Finch built limits into the Machine; Chappie's maker made him promise
"no crimes". Ours are about the people it watches, and they are enforced in
code, not just stated:

1. Nothing leaves the device. After the one-time model download there is no
   network code path (perception, memory, speech all run locally).
2. Embeddings, not pictures. Faces are stored as 128-d vectors; images are
   kept only if the owner turns on ``save_thumbnails``.
3. One admin. The first person the Machine is taught becomes its admin
   (the "creator"). Destructive commands (forget a person, wipe memory) only
   obey the admin, and only while the admin is in view.
4. Strangers are not catalogued forever. Unnamed strangers decay like any
   other episodic memory unless the admin names them.
5. Everyone can ask to be forgotten: the admin executes it, and it is a
   real delete (row + prototypes + episodes), not a flag.
"""
from __future__ import annotations

from ..memory.store import Store


class Ethics:
    def __init__(self, store: Store):
        self.store = store

    @property
    def admin_id(self) -> int | None:
        return self.store.get_self("admin_id")

    def set_admin(self, cid: int) -> None:
        self.store.set_self("admin_id", cid)
        self.store.x("UPDATE concepts SET role='admin' WHERE id=?", (cid,))

    def is_admin(self, cid: int | None) -> bool:
        return cid is not None and cid == self.admin_id

    def may_forget(self, requester: int | None) -> bool:
        return self.admin_id is None or self.is_admin(requester)

    def forget(self, cid: int) -> None:
        """Right to be forgotten: hard delete everything tied to a concept."""
        with self.store.transaction() as s:
            s._db.execute("DELETE FROM episodes WHERE concept_id=?", (cid,))
            s._db.execute("DELETE FROM concepts WHERE id=?", (cid,))   # cascades prototypes + lexicon
        if self.admin_id == cid:
            self.store.set_self("admin_id", None)
