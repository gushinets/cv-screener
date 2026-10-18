## Browser tests on Windows

Every Chromium launch, including one-off commands and document previews, must
include `--disable-features=AutofillAiWalletPrivatePasses`. Chrome 153/154 can
otherwise probe the Windows account with an empty password and lock it out.

- Playwright: pass `args=['--disable-features=AutofillAiWalletPrivatePasses']`.
- agent-browser: run from this project root to load `agent-browser.json`, or
  pass `--args '--disable-features=AutofillAiWalletPrivatePasses'` explicitly.
- Preserve the regression checks in `tests/test_browser_launch.py`.
- Do not change Windows passwords, account-lockout policies, or audit settings
  to make browser tests work.
