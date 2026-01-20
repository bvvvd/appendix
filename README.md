# Appendix

Appendix is a **self-study event sourcing toolkit** for Kotlin applications, built on top of PostgreSQL.

It was created as a hands-on learning project to explore **append-only data models**, **idempotent processing**, and **rebuildable read models** in real code, rather than through theoretical examples.

The focus is on understanding trade-offs, failure modes, and operational simplicity when building event-driven systems.

---

## Purpose

Appendix is primarily a **learning-by-building project**.

Its goals are to:
- deepen practical understanding of event sourcing concepts
- experiment with append-only persistence and optimistic concurrency
- explore projector design, checkpointing, and rebuildability
- study reliability patterns such as idempotency and at-least-once delivery
- reason about system behavior under retries and partial failures

It is intentionally **not positioned as a production-ready framework**.

---

## Why Appendix?

Many existing event sourcing solutions are:
- too complex for experimentation,
- tightly coupled to large infrastructures,
- or abstract away important details.

Appendix takes the opposite approach:
- minimal abstractions
- explicit data flow
- PostgreSQL as the only dependency
- clear and inspectable state

This makes it suitable for **learning, experimentation, and small internal tools**.

---

## Core Concepts

### Event Store
Events are stored in an append-only PostgreSQL table.

Each event contains:
- a globally ordered sequence number
- a stream-local version (for optimistic concurrency)
- immutable payload and metadata

### Streams
A stream represents a single aggregate instance (e.g. `Transaction-123`).

Concurrency control is achieved by appending events with an **expected stream version**.

### Projectors
Projectors consume events in order and build **read models**.

They:
- process events in batches
- maintain checkpoints
- can be safely restarted or rebuilt

### Idempotency
Appendix supports idempotent writes via optional idempotency keys.

This enables safe retries and experimentation with at-least-once delivery semantics.

---

## Design Principles

- **Learning over completeness**  
  The design favors clarity and inspectability over feature richness.

- **Append-only first**  
  Events are immutable; corrections are expressed as new events.

- **PostgreSQL as the source of truth**  
  No message brokers or distributed logs are required.

- **Rebuildability**  
  Read models are disposable and can be rebuilt from events at any time.

- **Explicit trade-offs**  
  Limitations are intentional and documented.

---

## Typical Learning Scenarios

Appendix is useful for:
- studying event sourcing in practice
- experimenting with projector patterns
- testing idempotent processing strategies
- building pet projects with auditability requirements
- exploring retry and replay behavior

It is **not intended for large-scale distributed systems**.

---

## High-level Architecture

****
