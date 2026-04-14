/**
 * ARX VR Teleop — Minimal interface helpers.
 * Desktop status panel removed; only VR status callback remains.
 */

let vrConnected = false;

function updateVRStatus(connected) {
  vrConnected = connected;
  console.log(`VR status: ${connected ? 'connected' : 'disconnected'}`);
}
