# Review Guidelines

- Kotlin + coroutines allowed.
- Processing is at-least-once; handlers must be idempotent.
- Postgres is source of truth; raw events are immutable; corrections create new versions.
- External calls must have timeouts + retries + backoff.