import { gcm } from "@noble/ciphers/aes.js";
import { randomBytes } from "@noble/ciphers/utils.js";
import { x25519 } from "@noble/curves/ed25519.js";
import { hkdf } from "@noble/hashes/hkdf.js";
import { hmac } from "@noble/hashes/hmac.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex, concatBytes } from "@noble/hashes/utils.js";

type DeviceCredential = { device_id: string; device_secret: string };
type Transfer = {
  transfer_id: string;
  kind: string;
  filename: string;
  mime_type: string;
  size: number;
};
type PhoneRequest = { request_id: string; text: string; status: string; created_at: number; result: string };
type RequestState = { accepting: boolean; requests: PhoneRequest[] };

const STATUS_WORDS: Record<string, string> = {
  waiting: "Waiting on your computer",
  running: "Running",
  done: "Done",
  failed: "Did not finish",
  dismissed: "Dismissed on the computer",
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`Air Handoff receiver is missing ${selector}`);
  return found;
}

const pairCard = element<HTMLElement>("#pair-card");
const receiveCard = element<HTMLElement>("#receive-card");
const pairButton = element<HTMLButtonElement>("#pair-button");
const forgetButton = element<HTMLButtonElement>("#forget-button");
const deviceName = element<HTMLInputElement>("#device-name");
const pairError = element<HTMLElement>("#pair-error");
const receiveError = element<HTMLElement>("#receive-error");
const connectionState = element<HTMLElement>("#connection-state");
const emptyState = element<HTMLElement>("#empty-state");
const transferList = element<HTMLElement>("#transfer-list");
const askCard = element<HTMLElement>("#ask-card");
const askText = element<HTMLTextAreaElement>("#ask-text");
const askButton = element<HTMLButtonElement>("#ask-button");
const askError = element<HTMLElement>("#ask-error");
const requestList = element<HTMLElement>("#request-list");

let credential: DeviceCredential | null = loadCredential();
let pollTimer = 0;
const renderedTransfers = new Set<string>();

deviceName.value = navigator.userAgent.includes("Android")
  ? "Android phone"
  : navigator.userAgent.includes("iPhone")
    ? "iPhone"
    : "My phone";

function b64u(bytes: Uint8Array): string {
  let binary = "";
  for (const value of bytes) binary += String.fromCharCode(value);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function unb64u(value: string): Uint8Array {
  const padded = value.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - (value.length % 4)) % 4);
  return Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

function loadCredential(): DeviceCredential | null {
  try {
    const parsed = JSON.parse(localStorage.getItem("heliox_air_handoff_credential") || "null");
    if (parsed?.device_id && parsed?.device_secret) return parsed as DeviceCredential;
  } catch {
    // Ignore corrupt local state and require pairing again.
  }
  return null;
}

function saveCredential(value: DeviceCredential): void {
  credential = value;
  localStorage.setItem("heliox_air_handoff_credential", JSON.stringify(value));
}

function forgetCredential(): void {
  credential = null;
  localStorage.removeItem("heliox_air_handoff_credential");
  renderedTransfers.clear();
  transferList.replaceChildren();
  showMode();
}

async function pairPhone(): Promise<void> {
  pairError.textContent = "";
  const fragment = new URLSearchParams(location.hash.slice(1));
  const encodedSecret = fragment.get("pair");
  if (!encodedSecret) {
    pairError.textContent = "Open this page by scanning the current pairing QR in Heliox.";
    return;
  }
  const name = deviceName.value.trim();
  if (!name) {
    pairError.textContent = "Enter a name for this phone.";
    return;
  }

  pairButton.disabled = true;
  pairButton.textContent = "Establishing encrypted channel…";
  try {
    const pairingSecret = unb64u(encodedSecret);
    const keyPair = x25519.keygen();
    const clientProof = hmac(sha256, pairingSecret, concatBytes(encoder.encode("pair-v1:"), keyPair.publicKey));
    const response = await fetch("/api/pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_name: name,
        client_public_key: b64u(keyPair.publicKey),
        client_proof: b64u(clientProof),
      }),
    });
    const payload = (await response.json()) as Record<string, string>;
    if (!response.ok) throw new Error(payload.error || "Pairing failed");

    const serverPublic = unb64u(payload.server_public_key);
    const expectedProof = hmac(
      sha256,
      pairingSecret,
      concatBytes(encoder.encode("server-v1:"), serverPublic, keyPair.publicKey),
    );
    if (!equalBytes(expectedProof, unb64u(payload.server_proof))) {
      throw new Error("The computer could not be authenticated");
    }
    const shared = x25519.getSharedSecret(keyPair.secretKey, serverPublic);
    const wrappingKey = hkdf(sha256, shared, pairingSecret, encoder.encode("heliox-air-handoff-pair-v1"), 32);
    const clear = gcm(wrappingKey, unb64u(payload.nonce), encoder.encode("heliox-air-handoff-credential-v1")).decrypt(
      unb64u(payload.credential),
    );
    saveCredential(JSON.parse(decoder.decode(clear)) as DeviceCredential);
    history.replaceState(null, "", location.pathname);
    showMode();
    await poll();
  } catch (error) {
    pairError.textContent = error instanceof Error ? error.message : "Pairing failed";
  } finally {
    pairButton.disabled = false;
    pairButton.textContent = "Pair securely";
  }
}

async function signedFetch(path: string, options: RequestInit = {}): Promise<Response> {
  if (!credential) throw new Error("This phone is not paired");
  const method = (options.method || "GET").toUpperCase();
  const body = typeof options.body === "string" ? encoder.encode(options.body) : new Uint8Array();
  const timestamp = String(Date.now() / 1000);
  const nonce = b64u(randomBytes(18));
  const canonical = encoder.encode([method, path, timestamp, nonce, bytesToHex(sha256(body))].join("\n"));
  const signature = hmac(sha256, unb64u(credential.device_secret), canonical);
  return fetch(path, {
    ...options,
    headers: {
      ...(options.headers || {}),
      "X-Heliox-Device": credential.device_id,
      "X-Heliox-Time": timestamp,
      "X-Heliox-Nonce": nonce,
      "X-Heliox-Signature": b64u(signature),
    },
  });
}

function decryptJson<T>(envelope: { nonce: string; ciphertext: string }, aad: string): T {
  if (!credential) throw new Error("This phone is not paired");
  const clear = gcm(unb64u(credential.device_secret), unb64u(envelope.nonce), encoder.encode(aad)).decrypt(
    unb64u(envelope.ciphertext),
  );
  return JSON.parse(decoder.decode(clear)) as T;
}

function encryptJson(payload: unknown, aad: string): { nonce: string; ciphertext: string } {
  if (!credential) throw new Error("This phone is not paired");
  const nonce = randomBytes(12);
  const sealed = gcm(unb64u(credential.device_secret), nonce, encoder.encode(aad)).encrypt(
    encoder.encode(JSON.stringify(payload)),
  );
  return { nonce: b64u(nonce), ciphertext: b64u(sealed) };
}

async function refreshRequests(): Promise<void> {
  const response = await signedFetch("/api/requests");
  const payload = (await response.json()) as { nonce: string; ciphertext: string; error?: string };
  if (!response.ok) throw new Error(payload.error || "Could not read your requests");
  const state = decryptJson<RequestState>(payload, "requests-v1");
  askCard.classList.toggle("hidden", !state.accepting && state.requests.length === 0);
  askButton.disabled = !state.accepting;
  askText.disabled = !state.accepting;
  if (!state.accepting) askError.textContent = "Requests are turned off on your computer.";
  else if (askError.textContent === "Requests are turned off on your computer.") askError.textContent = "";
  const items = state.requests.map((request) => {
    const item = document.createElement("li");
    item.textContent = request.text;
    const status = document.createElement("span");
    status.textContent = STATUS_WORDS[request.status] ?? request.status;
    item.append(status);
    if (request.result) {
      const answer = document.createElement("p");
      answer.className = "answer";
      answer.textContent = request.result;
      item.append(answer);
    }
    return item;
  });
  requestList.replaceChildren(...items);
}

async function sendRequest(): Promise<void> {
  const text = askText.value.trim();
  askError.textContent = "";
  if (!text) {
    askError.textContent = "Type what you want Heliox to do.";
    return;
  }
  askButton.disabled = true;
  try {
    const response = await signedFetch("/api/requests", {
      method: "POST",
      body: JSON.stringify(encryptJson({ text }, "request-v1")),
      headers: { "Content-Type": "application/json" },
    });
    const payload = (await response.json()) as { error?: string };
    if (!response.ok) throw new Error(payload.error || "The computer refused the request");
    askText.value = "";
    await refreshRequests();
  } catch (error) {
    askError.textContent = error instanceof Error ? error.message : "Could not send the request";
  } finally {
    askButton.disabled = false;
  }
}

async function poll(): Promise<void> {
  if (!credential) return;
  try {
    const response = await signedFetch("/api/pending");
    const payload = (await response.json()) as { nonce: string; ciphertext: string; error?: string };
    if (!response.ok) throw new Error(payload.error || "Receiver authentication failed");
    const pending = decryptJson<Transfer[]>(payload, "pending-v1");
    receiveError.textContent = "";
    for (const transfer of pending) {
      if (!renderedTransfers.has(transfer.transfer_id)) await receiveTransfer(transfer);
    }
    emptyState.classList.toggle("hidden", renderedTransfers.size > 0);
    await refreshRequests();
  } catch (error) {
    receiveError.textContent = error instanceof Error ? error.message : "Could not reach Heliox";
  } finally {
    clearTimeout(pollTimer);
    if (credential) pollTimer = window.setTimeout(poll, 1800);
  }
}

async function receiveTransfer(transfer: Transfer): Promise<void> {
  if (!credential) throw new Error("This phone is not paired");
  const path = `/api/transfers/${encodeURIComponent(transfer.transfer_id)}`;
  const response = await signedFetch(path);
  if (!response.ok) {
    const payload = (await response.json()) as { error?: string };
    throw new Error(payload.error || "Could not download the handoff");
  }
  const encrypted = new Uint8Array(await response.arrayBuffer());
  if (encrypted.length < 29) throw new Error("The encrypted handoff is incomplete");
  const clear = gcm(
    unb64u(credential.device_secret),
    encrypted.slice(0, 12),
    encoder.encode(`transfer-v1:${transfer.transfer_id}`),
  ).decrypt(encrypted.slice(12));
  const blob = new Blob([new Uint8Array(clear)], { type: transfer.mime_type });
  const url = URL.createObjectURL(blob);
  const text = transfer.mime_type === "text/plain" && clear.length <= 20000 ? decoder.decode(clear) : "";
  renderTransfer(transfer, url, text);
  renderedTransfers.add(transfer.transfer_id);
}

function renderTransfer(transfer: Transfer, url: string, text = ""): void {
  const article = document.createElement("article");
  article.className = "transfer";
  if (transfer.mime_type.startsWith("image/")) {
    const image = document.createElement("img");
    image.src = url;
    image.alt = transfer.filename;
    article.append(image);
  }
  const info = document.createElement("div");
  info.className = "transfer-info";
  const title = document.createElement("strong");
  title.textContent = transfer.filename;
  const details = document.createElement("span");
  details.textContent = `${transfer.kind} · ${formatBytes(transfer.size)}`;
  if (text) {
    const preview = document.createElement("p");
    preview.className = "answer";
    preview.textContent = text;
    info.append(preview);
  }
  const actions = document.createElement("div");
  actions.className = "transfer-actions";
  const download = document.createElement("a");
  download.href = url;
  download.download = transfer.filename;
  download.textContent = "Save";
  const done = document.createElement("button");
  done.type = "button";
  done.textContent = "Received";
  done.addEventListener("click", async () => {
    const path = `/api/transfers/${encodeURIComponent(transfer.transfer_id)}/ack`;
    const response = await signedFetch(path, {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" },
    });
    if (!response.ok) return;
    article.remove();
    URL.revokeObjectURL(url);
    renderedTransfers.delete(transfer.transfer_id);
    emptyState.classList.toggle("hidden", renderedTransfers.size > 0);
  });
  actions.append(download, done);
  info.append(title, details, actions);
  article.append(info);
  transferList.prepend(article);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function showMode(): void {
  const paired = Boolean(credential);
  pairCard.classList.toggle("hidden", paired);
  receiveCard.classList.toggle("hidden", !paired);
  if (!paired) askCard.classList.add("hidden");
  connectionState.classList.toggle("online", paired);
  connectionState.textContent = paired ? "Paired · encrypted" : "Not paired";
  clearTimeout(pollTimer);
  if (paired) void poll();
}

pairButton.addEventListener("click", () => void pairPhone());
askButton.addEventListener("click", () => void sendRequest());
forgetButton.addEventListener("click", forgetCredential);
showMode();
