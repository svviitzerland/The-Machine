"""Admin and forgetting rules.

The first person the Machine is taught becomes its admin, and only the admin, while in view,
can make it forget someone. Forgetting is a hard delete of the concept and its episodes.
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
