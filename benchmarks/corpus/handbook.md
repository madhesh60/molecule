# Northwind Parcel — Engineering Handbook

This handbook is the working agreement for everyone who ships code at Northwind
Parcel. It is long on purpose: an agent or a new hire should be able to answer
"how do we do X here?" from one section without reading the rest. When a rule
here disagrees with an old wiki page, this file wins.

## 1. Getting started

### 1.1 Local development setup

You need Docker Desktop, Node 20 and Python 3.12. Everything else is installed
by the bootstrap script.

```bash
# Clone and bootstrap (takes about six minutes on a cold cache)
git clone git@github.com:northwind-parcel/platform.git
cd platform
make bootstrap

# Start the full stack: api, worker, postgres, redis, opensearch
docker compose up -d

# Load demo shipments so the dashboard is not empty
make seed
```

Copy `.env.example` to `.env.local` before the first run. The API listens on
port 8420 and the dashboard on port 5173. If `make bootstrap` fails on Apple
Silicon, run `softwareupdate --install-rosetta` first.

### 1.2 Repository layout

The platform is a single monorepo. Top-level folders:

| Folder | What lives there |
|---|---|
| `services/api` | The public REST API (FastAPI) |
| `services/worker` | Background job consumers |
| `services/tracker` | Carrier polling and tracking-event ingestion |
| `packages/ui` | The React dashboard and the shared component library |
| `infra/terraform` | All cloud infrastructure, one folder per environment |
| `docs/adr` | Architecture decision records |

Shared Python code goes in `libs/`, never copy-pasted between services. A new
top-level folder needs an ADR (see §13.2).

### 1.3 Who to ask

Each area has an owning team. Ask in the team channel before you DM a person.

| Area | Team | Channel |
|---|---|---|
| Public API, webhooks | Integrations | `#team-integrations` |
| Tracking pipeline | Signals | `#team-signals` |
| Dashboard, mobile | Surfaces | `#team-surfaces` |
| Infra, CI, deploys | Platform | `#team-platform` |
| Billing, invoices | Ledger | `#team-ledger` |

Questions with no obvious owner go to `#eng-help`, which the Platform team
triages every weekday morning.

## 2. Code workflow

### 2.1 Branching strategy

We practice trunk-based development. Branch from `main`, keep the branch alive
for at most two working days, and merge back through a pull request. Branch
names use a prefix: `feat/`, `fix/`, `chore/` or `spike/`. There are no
long-lived release branches; a release is just a tagged commit on `main`.
Spike branches are never merged — they are deleted after the write-up.

### 2.2 Pull requests

Keep a pull request under 400 changed lines; larger changes are split into a
stack. Every PR needs one approval, except anything touching
`services/api/billing`, which needs two approvals including one from the
Ledger team. We always squash-merge so `main` stays one commit per change.
Fill in the PR template: what changed, why, how it was tested, and the rollout
plan. Draft PRs are welcome for early feedback and do not page reviewers.

### 2.3 Commit messages

Commit subjects follow Conventional Commits: `type(scope): summary`, written in
the imperative mood and capped at 72 characters. Allowed types are `feat`,
`fix`, `refactor`, `perf`, `test`, `docs`, `chore` and `revert`. The body
explains why, not what. Reference the ticket as a trailer, for example
`Refs: NP-4821`. The changelog is generated from these subjects, so a sloppy
subject becomes a sloppy release note.

### 2.4 Code review etiquette

Reviewers respond within one business day; if you cannot, say so and hand the
review to someone else. Prefix optional suggestions with `nit:` so the author
knows they are not blocking. Review the change, not the person. Authors resolve
their own threads after addressing them. Approving with open questions is fine
when you trust the author to follow up — say "approve with comments".

## 3. Testing

### 3.1 Unit tests

Python services use pytest; the dashboard uses Vitest. Unit tests must not touch
the network, the clock or the filesystem outside a temp directory. The coverage
floor is 85% on changed lines, enforced in CI by `diff-cover`. Do not chase the
number with assertion-free tests — a reviewer will ask you to delete them.

### 3.2 Integration tests

Integration tests run against real dependencies started by Testcontainers:
Postgres 16, Redis 7 and a local OpenSearch node. They live next to the service
under `tests/integration` and run in the merge queue, not on every push. Each
test owns its data: create what you need in a fixture and never depend on rows
left behind by another test. Mark slow cases with `@pytest.mark.slow`.

### 3.3 End-to-end tests

End-to-end tests are written in Playwright and run nightly against staging,
plus on demand with the `e2e` label on a PR. A test that fails three times in
seven days without a code change is quarantined automatically: it is moved to
the `quarantine` project, a ticket is opened for the owning team, and it stops
blocking deploys. Quarantined tests must be fixed or deleted within two sprints.

### 3.4 Load testing

We load test with k6 before every major launch and before the peak season
freeze. The pass bar is p95 latency under 250 ms at twice the expected peak
traffic, sustained for 20 minutes, with an error rate below 0.1%. Scripts live
in `perf/k6`. Never point a load test at production; use the `perf` environment,
which is a full-size copy of production that Platform scales up on request.

## 4. API design

### 4.1 Versioning

The public API is versioned in the URL path: `/v1/...`, `/v2/...`. Additive
changes (new optional fields, new endpoints) ship inside the current version.
Breaking changes need a new major version and a migration guide. An old version
is deprecated with a `Sunset` response header and stays alive for six months
after the announcement. Internal service-to-service endpoints are unversioned.

### 4.2 Pagination

List endpoints use cursor-based pagination, never page numbers or offsets. The
client passes `limit` (default 50, maximum 200) and the opaque `cursor` from the
previous response. Responses include `next_cursor`, which is null on the last
page. Cursors are valid for one hour. Sort order must be stable, so always break
ties on the primary key.

### 4.3 Error format

Errors are returned as `application/problem+json` following RFC 7807. Every
error body carries `type`, `title`, `status`, `detail` and our own `code` field.

| Code | HTTP status | Meaning |
|---|---|---|
| `validation_failed` | 422 | A field failed validation; see `errors[]` |
| `not_found` | 404 | The resource does not exist or is not visible to you |
| `conflict` | 409 | The request conflicts with current state |
| `rate_limited` | 429 | Too many requests; retry after the given delay |
| `internal` | 500 | Our fault; includes a `trace_id` to quote to support |

Never leak stack traces or SQL in `detail`.

### 4.4 Idempotency

Every mutating endpoint accepts an `Idempotency-Key` header. The first response
for a key is stored in Redis for 24 hours and replayed for any retry with the
same key and the same request body. Reusing a key with a different body returns
409 `conflict`. Clients should generate a UUID per logical operation. This is
what makes it safe for customers to retry a timed-out "create shipment" call.

### 4.5 Rate limiting

Rate limits use a token bucket per API key: 600 requests per minute with a burst
of 100. Over the limit we return 429 with a `Retry-After` header in seconds.
Bulk endpoints cost ten tokens per call. Limits are enforced at the edge gateway,
so a service behind it must not implement its own throttling. Enterprise plans
can request a higher ceiling through their account manager.

### 4.6 Webhooks

Outgoing webhooks are signed with HMAC-SHA256 using the endpoint's signing
secret; the signature goes in the `X-Northwind-Signature` header along with a
timestamp, and receivers should reject anything older than five minutes. A
delivery that does not get a 2xx within ten seconds is retried with exponential
backoff for up to 72 hours, ten attempts in total. After the last failure the
endpoint is disabled and the account owner is emailed. Deliveries follow the
same idempotency rules as §4.4: each event has a stable `event_id`.

## 5. Data

### 5.1 Database migrations

Schema changes use Alembic and the expand/contract pattern: first add the new
column or table (expand), deploy code that writes both, backfill, then remove
the old shape in a later deploy (contract). Never ship a destructive migration
in the same deploy as the code that stops using the column. Every migration
sets `lock_timeout` to 5 seconds so a blocked DDL fails fast instead of queueing
behind traffic. Backfills run as batched background jobs, not inside the
migration.

### 5.2 Backups and restore

Postgres is backed up with nightly `pg_dump` snapshots plus continuous WAL
archiving, which gives point-in-time recovery for the last 14 days. Backups are
encrypted and copied to a second region. A backup you have never restored is a
hope, not a backup: Platform runs a full restore drill every quarter and
records the time it took. The current recovery time objective is four hours.

### 5.3 Data retention

Tracking events are kept for 18 months and then deleted by a weekly job. When
an account is closed, personal data is purged within 30 days; invoices are kept
for seven years because tax law requires it. Application logs are retained for
45 days. Never copy production data to a laptop — use the anonymised snapshot
that the `make seed-realistic` target downloads.

### 5.4 Caching

Redis is the only cache. Use the cache-aside pattern: read the cache, fall back
to the database on a miss, then populate the cache. The default TTL is five
minutes; anything longer needs a comment explaining why stale data is safe.
Keys are named `service:entity:id`, for example `api:shipment:8f2c`. Invalidate
on write rather than waiting for expiry. Never cache anything keyed only by
user input without a length limit.

### 5.5 Search indexing

Shipment search is served by OpenSearch. Documents are written by the
`indexer` worker from the outbox table, so search can lag the database by a few
seconds. A mapping change requires a full reindex: build the new index beside
the old one, then switch the read alias in one atomic step (a blue/green alias
swap). Keep the old index for 24 hours in case the switch has to be reversed.

## 6. Security

### 6.1 Secrets management

All secrets live in Vault. Services read them at start-up through the sidecar;
nothing secret is ever committed, pasted in chat or baked into an image. Local
development uses throwaway values from `.env.example`. Production credentials
are rotated every 90 days by an automated job, and immediately when someone
with access leaves. If you find a secret in the repository, treat it as leaked:
rotate it first, then remove it from history.

### 6.2 Authentication

Users sign in through OAuth 2.0 with PKCE. The API issues a short-lived JWT
access token that expires after 15 minutes and a refresh token that lasts 30
days. Refresh tokens are rotated on every use, and reuse of an old refresh token
revokes the whole session family. Machine clients use API keys, which are
stored hashed and shown to the customer exactly once at creation.

### 6.3 Authorization

Access control is role-based. There are three roles: `viewer` can read
shipments, `dispatcher` can create and edit shipments, and `admin` can manage
users, API keys and billing. Checks are done in the `authz` middleware using
policy functions — never by comparing role strings inside a handler. Every
query is scoped by `tenant_id`; a missing tenant filter is treated as a
security bug, not a style issue.

### 6.4 Dependency updates

Renovate opens dependency update PRs every Monday. Patch and minor bumps with
green CI are auto-merged; majors need a human. A critical security advisory
must be patched within 48 hours, high within seven days. Do not pin around a
vulnerability without a ticket and an expiry date on the pin. Base container
images are rebuilt weekly so OS packages do not go stale.

### 6.5 Reporting a security incident

If you suspect a breach or find a vulnerability, email
`security@northwindparcel.example` or page the security on-call. Do not discuss
details in public channels and do not try to "confirm" the issue by exploring
further. Security acknowledges within one hour. Customer-facing communication
about a security incident is written by Security and Legal only.

## 7. Deploy and release

### 7.1 CI pipeline

CI runs on GitHub Actions in four stages: lint, unit tests, build, and
integration tests. The whole pipeline must finish in under 12 minutes; if your
change makes it slower, fix that in the same PR. Lint failures do not start the
later stages. Build artifacts are container images tagged with the commit SHA,
and the same image is promoted through every environment — we never rebuild for
production.

### 7.2 Deploy process

Every merge to `main` is deployed to staging automatically. Production deploys
are started from chat with the `/deploy <service>` command and go out as a
canary first: 5% of traffic for 15 minutes while the deploy bot watches error
rate and latency. If the canary is healthy, the rollout continues to 100%
without another prompt. You deploy your own change and you stay online until
it is fully rolled out. If anything looks wrong, roll back first and
investigate afterwards (see §7.3).

### 7.3 Rollback

To roll back a service, run:

```bash
# Reverts the service to the previous healthy image
deployctl rollback <service>

# Check that the old version is serving
deployctl status <service>
```

A rollback completes in under five minutes and does not need approval. Database
changes are never rolled back: because migrations are expand/contract (§5.1),
the previous code version always works with the current schema, and any data
fix is done as a forward fix.

### 7.4 Feature flags

Risky changes ship dark behind a feature flag in Flagsmith. Every flag has an
owner and an expiry date in its description. Roll out by percentage or by
tenant, and keep the flag evaluation out of hot loops. Once a flag has been at
100% for 30 days it must be removed from the code; a weekly bot posts stale
flags to the owning team's channel. Flags are for rollout, not for permanent
configuration — use settings for that.

### 7.5 Release freeze

There are no production deploys on Fridays after 15:00 local time. During the
peak season freeze, from 20 November to 31 December, only fixes for SEV1 and
SEV2 incidents may ship, and each one needs approval from the VP of
Engineering. Staging stays open during a freeze. Plan large launches to land at
least two weeks before the freeze starts.

### 7.6 Mobile releases

The mobile apps ship on a biweekly release train. The release branch is cut on
Tuesday, tested for two days, and submitted to the stores on Thursday. Rollout
is phased: 1%, then 10%, then 50%, then 100%, with at least 24 hours at each
step and a crash-free sessions bar of 99.5% to continue. A change that misses
the cut waits for the next train; there are no "quick" store submissions.

## 8. Observability

### 8.1 Logging

Logs are structured JSON, one event per line. Every log line carries the
`x-request-id` correlation id so a request can be followed across services.
Never log personal data, tokens or full request bodies. Use `info` for business
events, `warning` for things that recovered on their own, and `error` only when
a human should eventually look. Debug logging is off in production and can be
enabled per tenant for one hour.

### 8.2 Metrics

Services export Prometheus metrics. Instrument every endpoint with the RED
method: rate, errors and duration. Metric names follow
`<service>_<noun>_<unit>`, for example `api_request_duration_seconds`. Keep
label cardinality low: never use a user id, shipment id or raw URL path as a
label. Dashboards live in Grafana under one folder per team.

### 8.3 Tracing

Distributed tracing uses OpenTelemetry. We sample 10% of normal requests and
100% of requests that end in an error. Propagate the trace context on every
outgoing HTTP call and on every queue message. Name spans after the operation,
not the URL. If a trace is missing spans, the usual cause is a library that was
not auto-instrumented.

### 8.4 Alerting

We page a human only when a user-facing SLO is burning (see §8.5); everything
else goes to a ticket or a channel. Every alert must link to a runbook that says
what to check first. An alert that pages twice without any action taken is a
bad alert and gets fixed or deleted that week. Alerts are defined in code under
`infra/alerts` and reviewed like any other change.

### 8.5 Service level objectives

| Service | Objective | Window |
|---|---|---|
| Public API availability | 99.9% of requests succeed | 30 days |
| Tracking lookup latency | p95 under 300 ms | 30 days |
| Webhook delivery | 99% delivered within 60 seconds | 7 days |

Each SLO has an error budget. When more than half of the budget is spent in a
window, the owning team pauses feature work for that service and spends the
time on reliability until the budget recovers.

## 9. Incidents and on-call

### 9.1 On-call rotation

Each team runs a weekly on-call rotation with a primary and a secondary. The
handoff happens on Monday at 10:00 with a short written summary of open issues.
The primary acknowledges a page within 10 minutes; if they do not, it escalates
to the secondary. Anyone who is paged outside working hours takes a
compensation day the same week. Swaps are fine — update the schedule yourself.

### 9.2 Incident severity levels

| Level | Definition | Example |
|---|---|---|
| SEV1 | Most customers cannot use a core feature | API down, tracking pages blank |
| SEV2 | A core feature is degraded or one large customer is blocked | Webhooks delayed by an hour |
| SEV3 | A minor feature is broken with a workaround | CSV export times out |
| SEV4 | Cosmetic or internal-only | Typo in an email template |

When in doubt, pick the higher severity; it is cheap to downgrade later.

### 9.3 Incident response

Declare an incident in `#incidents` with the `/incident` command the moment you
suspect customer impact. Three roles are assigned immediately: the incident
commander makes decisions, the communications lead owns updates, and the scribe
keeps the timeline. For a SEV1 the public status page is updated within 15
minutes and then every 30 minutes until resolved. Mitigate first — roll back,
fail over or turn off the flag — and look for the root cause afterwards.

### 9.4 Postmortems

Every SEV1 and SEV2 gets a blameless postmortem, published within five business
days. It covers the timeline, the impact, what went well, what went badly, and
action items with an owner and a due date. Action items are tracked in the
`POSTMORTEM` project and reviewed in the monthly reliability meeting. We write
about systems and decisions, never about who made a mistake.

## 10. Frontend

### 10.1 Component library

The dashboard is React and TypeScript. Shared components live in `packages/ui`
and are documented in Storybook; build a screen from existing components before
writing a new one. Styling uses design tokens for colour, spacing and type — no
hard-coded hex values. A new shared component needs a story, a visual
regression snapshot and a short usage note.

### 10.2 Accessibility

We target WCAG 2.1 level AA. Every interactive element must be reachable and
usable with the keyboard alone, with a visible focus ring. The `axe` checker
runs in CI on every story and a new violation fails the build. Colour is never
the only way to convey state. Test new flows with a screen reader before
calling them done.

### 10.3 Performance budgets

The dashboard's initial JavaScript bundle must stay at or below 180 KB gzipped,
and the largest contentful paint must be under 2.5 seconds on a mid-range
phone over 4G. CI compares the bundle size against `main` and fails the build
on a regression of more than 5 KB. Lazy-load routes and heavy charts instead of
raising the budget.

### 10.4 Internationalization

All user-visible text goes through the translation layer using ICU message
format. Never build a sentence by concatenating strings, because word order
differs between languages. The product ships in nine locales; a new string is
merged in English and translated before the next mobile train. Dates, numbers
and currency are formatted with the `Intl` APIs, not by hand.

## 11. Background jobs

### 11.1 Queues

Asynchronous work goes through Amazon SQS, one queue per job type. Handlers
must be idempotent because SQS delivers at least once. The visibility timeout
is set to six times the handler's p99 runtime. Put the smallest possible
payload on the queue — an id, not the whole record — and load fresh data inside
the handler.

### 11.2 Retries and dead letters

A failing job is retried up to five times with exponential backoff and jitter.
After the fifth failure the message moves to that queue's dead-letter queue.
The owning team reviews its dead-letter queue every working day and either
replays the messages after a fix or discards them with a note. Never retry on
validation errors — they will fail the same way every time.

### 11.3 Scheduled jobs

Recurring work runs as Kubernetes CronJobs defined in `infra/cron`. All
schedules are written in UTC. A scheduled job must be safe to run twice and
safe to skip once, because the scheduler guarantees neither. Jobs that run
longer than their interval need a lock. Each job emits a heartbeat metric so a
silent failure raises an alert.

## 12. Cost and performance

### 12.1 Cloud cost guardrails

Every cloud resource is tagged with `team`, `service` and `env`; untagged
resources are deleted after a warning. Each team has a monthly budget, and an
alert fires when spend reaches 80% of it. New infrastructure that will cost
more than 500 dollars a month needs a line in the PR description saying so.
Turn off preview environments after three days of inactivity.

### 12.2 Query performance

No N+1 queries: load related rows with a join or a batched lookup. Any query
slower than 100 ms in production shows up in the slow-query dashboard; run
`EXPLAIN ANALYZE` on it and add an index or rewrite it. New indexes on large
tables are created with `CONCURRENTLY`. Do not paginate with `OFFSET` on tables
over a million rows (see §4.2).

## 13. People and process

### 13.1 Onboarding checklist

In your first week you should: get the stack running locally (§1.1), ship one
small change to production with your onboarding buddy, shadow one on-call
handoff, and read the last three postmortems. Your manager books a 30-minute
architecture walkthrough with each team. You join the on-call rotation only
after your second month.

### 13.2 Architecture decision records

A decision that is expensive to reverse gets an ADR in `docs/adr`. Use the
template: context, decision, alternatives considered, consequences. An ADR is
proposed as a pull request and is accepted when two staff engineers approve.
ADRs are never edited after acceptance; a later decision supersedes an earlier
one with a new record that links back.

### 13.3 Tech debt policy

Each team reserves 20% of every sprint for tech debt and reliability work. Debt
is tracked as normal tickets with the `debt` label and a one-line note on what
it costs us today. If you touch a file, leave it a little better than you found
it, but keep the clean-up in a separate commit from the behaviour change.
