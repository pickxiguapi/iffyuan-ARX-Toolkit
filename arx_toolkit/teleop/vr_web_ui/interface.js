/**
 * ARX VR Teleop — Desktop interface (status display).
 */

let vrConnected = false;

function updateVRStatus(connected) {
  vrConnected = connected;
  console.log(`VR status: ${connected ? 'connected' : 'disconnected'}`);

  const dot = document.getElementById('vrStatusDot');
  const txt = document.getElementById('vrStatusText');
  if (dot) {
    dot.style.background = connected ? '#0cce6b' : '#e94560';
  }
  if (txt) {
    txt.textContent = connected ? 'VR connected' : 'VR not connected';
  }
}
