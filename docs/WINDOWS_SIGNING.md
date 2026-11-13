# Windows code signing

Heliox OS uses the SignPath Foundation program for Windows Authenticode
signing. The integration deliberately separates test signing from production
release signing so a self-signed test certificate can never be presented to
users as a trusted release certificate.

## SignPath project

- Organization: `HelioxOS [OSS]`
- Project: `Heliox-OS`
- Artifact configuration: `windows-installers`
- Test policy: `test-signing`
- Production policy: `release-signing`

The `windows-installers` artifact configuration accepts the ZIP produced by
GitHub Actions and signs all three relevant layers:

1. the NSIS setup executable;
2. the MSI package; and
3. the Heliox executable embedded in the MSI.

## Test-signing workflow

Run **SignPath Windows test signing** manually from GitHub Actions. It:

1. builds one Windows EXE and one MSI on a GitHub-hosted runner;
2. uploads the unsigned pair as a short-lived GitHub Actions artifact;
3. submits that exact artifact to SignPath using the `test-signing` policy;
4. downloads the signed result;
5. fails unless both installers contain Authenticode signatures; and
6. uploads the signed test pair for inspection for seven days.

The workflow has read-only repository permissions and contains no release or
publication step. The API token is stored only as the repository secret
`SIGNPATH_API_TOKEN`.

The test certificate is self-signed. Windows may therefore report it as
untrusted even when the Authenticode signature is structurally present. Test
artifacts are for SignPath setup validation only and must not be distributed as
a Heliox release.

## Production cutover

The release workflow is ready for production signing and waits for one
switch: the repository variable `WINDOWS_RELEASE_SIGNING` set to `signpath`
(Settings > Secrets and variables > Actions > Variables). Set it only when all
of the following are true:

- the SignPath GitHub App is installed with access limited to `Heliox-OS`;
- the test-signing workflow completes successfully (last: 12 August, run
  31629964721);
- SignPath has reviewed the setup; and
- the Foundation production certificate is issued and the `release-signing`
  policy is valid, as the owner's SignPath account shows.

**While the variable is unset, or any other value**, releases are built and
published as before, with unsigned Windows installers.

**While it is `signpath`**, the release workflow:

1. builds the Windows EXE and MSI without attaching them to the release;
2. submits that exact pair to SignPath with the `release-signing` policy and
   the `windows-installers` artifact configuration (the same organization,
   project and action as the test workflow), and downloads the signed result;
3. checks that the setup program, the MSI and the application inside the MSI
   each carry a signature Windows reports as `Valid` (it chains to a trusted
   root; a missing, self-signed or untrusted signature fails), signed by the
   SignPath Foundation, with a timestamp so it outlives the certificate;
4. attaches only the signed pair, and refuses a release that already holds
   Windows installers it did not attach; and
5. publishes the release only after all of that succeeded.

It fails closed: if SignPath does not sign (the policy is not valid yet, the
review is not finished, the token is missing) or any check fails, the job
stops with a sentence saying why, nothing is attached for Windows and the
release is not published. There is no fallback to unsigned installers.
`daemon/tests/test_release_preflight.py` evaluates the workflow's own
conditions for both settings.

The application inside the NSIS setup program is not one of the three layers
the `windows-installers` configuration signs; the job reports its status as a
warning, not a failure. Signing it too would mean signing the executable before
NSIS packs it (a Tauri sign command calling SignPath), a separate change.

The `.msi` stays a release asset for now; its uninstall still leaves the daemon
running, and whether it stays is the owner's decision (FIRST_RUN.md).

The production certificate and private key remain in SignPath. They must not
be exported into GitHub secrets or committed to this repository.
