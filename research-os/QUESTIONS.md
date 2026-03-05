# DreamHub - Open Questions

Rolling log of open questions. Resolved items go to SPEC.md (decisions) and CHANGELOG.md (what was built).

---

1. **HF sync frequency**: Currently READMEs are fetched for ALL datasets during every sync. Should we only fetch for new/recently modified datasets? More broadly — does a full sync need to happen at every login, or should it be manual-only after the first load?

2. **Multi-user support**: Currently single-user (all authenticated users share data). If team members are added later, RLS policies would need user_id scoping. Deferred — not important now.
