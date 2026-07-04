## Sandbox Threat Model (read this before deploying with untrusted input)

`utils/sandbox.py` executes agent-submitted code using three layered controls:

1. **AST allow-list** (`_validate_ast`) — statically rejects imports outside a
   fixed safe-module list, and rejects direct `exec`/`eval`/`compile` calls.
2. **Restricted builtins** — the exec namespace only exposes a curated
   `__builtins__` dict (no `open`, no `__import__` of arbitrary modules at
   runtime beyond the allow-list).
3. **OS-level isolation** — code runs in a separate `multiprocessing.Process`
   (not just a thread), with `resource.setrlimit` caps on CPU time and file
   size (Linux/macOS). A virtual-memory cap (`RLIMIT_AS`) was attempted and
   removed: this process tree imports torch for the optional reward MLP,
   so the interpreter already occupies a large, version-dependent chunk of
   address space before the cap is even applied, and a size chosen without
   accounting for that intermittently broke thread creation for the
   `multiprocessing.Queue` feeder thread. See `utils/sandbox.py` for the
   full writeup and the two real fixes (cgroup memory limit at the
   container level, or decoupling the reward-model process from the
   execution sandbox). Fork-bomb protection (`RLIMIT_NPROC`) was similarly
   attempted and removed — that limit is per-UID system-wide, not
   per-process-tree, so it starves unrelated concurrent executions under
   load. Use a cgroup `pids.max` limit at the container level instead.

**What this does NOT protect against:**
- A sufficiently creative payload reaching unrestricted internals via
  attribute-chain traversal (e.g. walking from an allowed object through
  `__class__.__mro__` / `__subclasses__` back to something dangerous). This
  is a known, documented weakness of Python-level sandboxing in general —
  it is not unique to this project, but it is real and not fully closed here.
- Kernel/syscall-level attacks — there is no seccomp filter, no network
  namespace isolation, and the child process runs as the same OS user as
  the parent.
- Timing/resource side channels between sequential test runs.

**Recommended hardening for production / adversarial deployments:**
Run the executor inside a container with a read-only root filesystem, a
dropped-capabilities profile, `--network=none`, and a seccomp profile
(Docker + seccomp, or gVisor/Firecracker for stronger isolation). This
project intentionally documents the gap rather than claiming a false sense
of security — see `ARCHITECTURE.md` for the full threat-model writeup.

## Reporting a Vulnerability

We take the security of **CodeFix-Env** seriously. If you discover a security vulnerability, we appreciate your responsible disclosure and will work quickly to address it.

### How to Report

**Preferred Method:**  
Open a new discussion in the [GitHub Discussions](https://github.com/dhakarshailendra829/codefix-env/discussions) section of this repository.

- Please use the **Private vulnerability report** option if available, or clearly mark the discussion as sensitive.

> **Important**: Do not create public issues for security vulnerabilities. Use Discussions or email instead.

When submitting a report, please provide the following information:

- Clear description of the vulnerability
- Steps to reproduce the issue
- Affected version(s)
- Potential impact
- Any suggested mitigation or fix (if known)

### What to Expect

- You will receive an acknowledgment within **48–72 hours**.
- We will provide regular updates on the status of your report.
- We aim to resolve confirmed vulnerabilities as quickly as possible.

### Disclosure Policy

- We follow responsible disclosure practices.
- Please do **not** publicly disclose the vulnerability until a fix has been released (typically within 90 days).
- Once resolved, we will publish a GitHub Security Advisory and credit you (unless you prefer to remain anonymous).

### Scope

**In Scope:**
- Sandbox or container escapes
- Remote Code Execution (RCE)
- Authentication / Authorization bypass
- Sensitive data exposure
- Denial of Service affecting the host

**Out of Scope:**
- Issues found only in example code or documentation
- Attacks requiring physical or admin access
- Vulnerabilities in downstream LLMs

---

Thank you for helping keep **CodeFix-Env** secure! 🙏
