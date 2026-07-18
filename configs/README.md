# Configuration

`rtcache/` and `database/` contain unchanged configuration snapshots. Runtime
secrets and machine-specific paths must be supplied through environment variables
or CLI arguments. The organizer copies `.env.example` but never a real `.env`.
