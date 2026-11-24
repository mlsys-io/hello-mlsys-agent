// A fresh student's walk through the course page with the REAL configured models (not part of CI).
//   1. scripts/setup.sh; export QWEN38_* and MLAB_TEACHER_*; mlab web COURSE --no-browser --port 8799
//   2. (cd frontend && npm ci && npx playwright install chromium --only-shell)
//   3. node scripts/e2e/fresh-student.mjs [session] [steps.json]   (BASE_URL overrides http://127.0.0.1:8799)
// Each step types a message, approves the in-chat confirmation cards like the student would, waits for the
// agent to go idle, and prints the agent's replies and tool calls; screenshots land in E2E_OUT (default /tmp/e2e).
import { chromium } from '../../frontend/node_modules/playwright/index.mjs';
import { mkdirSync, readFileSync } from 'node:fs';
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8799';
const OUT = process.env.E2E_OUT || '/tmp/e2e';
mkdirSync(OUT, { recursive: true });
const DEFAULT_STEPS = [
  { say: 'Hi, I am new. In two sentences: what limits goodput in this task?', timeout: 600 },
  { say: 'Please draft an experiment named "Prefill chunking" with the baseline example and a variant chunk512 where serving.chunked_prefill_tokens is 512, simulate it, and tell me what happened to TTFT p99.', timeout: 1200, approve: true },
  { say: 'Please ask the teacher this question for me, exactly: "Why would chunked prefill make TTFT p99 worse in my simulation instead of better?"', timeout: 1200, approve: true },
  { say: 'Please arrange one sandbox measurement of the baseline configuration from r001 on the development workload (1 run) so we can compare it with the simulator.', timeout: 1200, approve: true },
];
let session = process.argv[2];  // default: the session the page opens
const steps = process.argv[3] ? JSON.parse(readFileSync(process.argv[3], 'utf8')) : DEFAULT_STEPS;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: 'en-US' });
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => m.type() === 'error' && errors.push('console: ' + m.text()));
const api = async (path) => (await (await fetch(BASE + path)).json());
const status = async () => (await api('/api/course')).sessions.find((s) => s.id === session)?.agent;
const chat = async () => (await api(`/api/sessions/${session}/chat`)).entries;
await page.goto(`${BASE}/`);  // opening the page starts a session for a new course
await page.waitForTimeout(1500);
session = session || new URL(page.url()).hash.split('/')[1];
if (!(await api('/api/course')).sessions.some((s) => s.id === session)) throw new Error(`No session ${session}`);
await page.goto(`${BASE}/#/${session}/overview`);
await page.waitForTimeout(1500);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
for (const [index, step] of steps.entries()) {
  const before = (await chat()).length;
  const started = Date.now();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill(step.say);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  log(`STEP ${index + 1} sent: ${step.say}`);
  let approved = 0;
  for (;;) {
    await page.waitForTimeout(3000);
    const cards = page.locator('.confirmation-card, .card-actions, [data-card-kind]');
    // Approve any in-chat confirmation, as the student would after reading it.
    for (const label of step.approve ? ['Run measurement', 'Ask the teacher', 'Allow', 'Approve', 'Confirm', 'Continue', 'Yes', 'Confirm and run'] : []) {
      const button = page.getByRole('button', { name: label, exact: true });
      if (await button.count() && await button.first().isVisible() && await button.first().isEnabled()) {
        await page.screenshot({ path: `${OUT}/step${index + 1}-card-${approved}.png`, fullPage: true });
        await button.first().click(); approved++; log(`  approved card via "${label}"`); break;
      }
    }
    const state = await status();
    if (state === 'waiting_approval' && !approved && Date.now() - started > 45000 && !step.dumped) {
      step.dumped = true;
      const labels = (await page.locator('.conversation-panel').getByRole('button').allInnerTexts()).map((t) => t.trim()).filter(Boolean);
      log('  UNRECOGNISED CARD; buttons:', JSON.stringify(labels));
      await page.screenshot({ path: `${OUT}/step${index + 1}-unknown-card.png` });
    }
    const entries = await chat();
    const fresh = entries.slice(before);
    if (state === 'idle' && fresh.some((e) => e.role === 'agent')) break;
    if (state === 'error' || Date.now() - started > (step.timeout || 900) * 1000) { log('  STOP state=', state); break; }
  }
  const entries = (await chat()).slice(before);
  for (const e of entries) {
    if (e.role === 'agent') log(`  AGENT: ${String(e.text).slice(0, 900).replace(/\n+/g, ' ⏎ ')}`);
    else if (e.role === 'tool' || e.tool) log(`  TOOL ${e.tool || ''} ${e.action || ''} ${e.is_error ? 'ERROR' : 'ok'}: ${String(e.result || e.text || '').slice(0, 200).replace(/\n+/g, ' ')}`);
    else if (e.role !== 'student') log(`  ${e.role.toUpperCase()}: ${String(e.text || '').slice(0, 300)}`);
  }
  log(`  took ${Math.round((Date.now() - started) / 1000)} s, status ${await status()}`);
  await page.screenshot({ path: `${OUT}/step${index + 1}.png`, fullPage: true });
}
const view = await api(`/api/workbench/sessions/${session}`);
log('RUNS', JSON.stringify((view.runs || []).map((r) => [r.id, r.status, r.verdict, r.candidate || r.candidate_id])));
log('EXPERIMENTS', JSON.stringify((view.experiments || []).map((x) => [x.id, x.title, x.author, (x.candidates || []).length])));
log('ERRORS', JSON.stringify(errors));
await browser.close();
