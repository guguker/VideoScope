# Security policy

VideoScope has no supported release series yet. Please report vulnerabilities
privately through GitHub's **Report a vulnerability** / Security Advisory flow for
this repository. Do not include credentials, private videos, database contents or
other personal data in a public issue.

The primary application is designed to bind only to loopback. Running it on a LAN
or public interface is unsupported without a separate authentication, TLS, body
limit and storage-quota design. The optional InternVideo service is a distinct
authenticated private-network boundary; follow `deploy/internvideo/README.md`.

Useful reports include the affected commit, endpoint or file format, a minimal
reproduction without private media, impact, and any proposed mitigation. Never
test against systems or data you do not own or have permission to assess.
