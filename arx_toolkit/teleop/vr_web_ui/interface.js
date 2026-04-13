/**
 * ARX VR Teleop — Desktop interface (status display only).
 * No keyboard control or settings — pure VR status monitor.
 */

let vrConnected = false;

function updateVRStatus(connected) {
  vrConnected = connected;
  const vrIndicator = document.getElementById('vrStatus');
  if (vrIndicator) {
    vrIndicator.className = 'status-indicator' + (connected ? ' connected' : '');
  }
}

function updateGripStatus(leftGrip, rightGrip) {
  const leftEl = document.getElementById('leftGripStatus');
  const rightEl = document.getElementById('rightGripStatus');
  if (leftEl) leftEl.className = 'status-indicator' + (leftGrip ? ' connected' : '');
  if (rightEl) rightEl.className = 'status-indicator' + (rightGrip ? ' connected' : '');
}

function updateUIForDevice() {
  const desktopInterface = document.getElementById('desktopInterface');
  const vrContent = document.getElementById('vrContent');

  if (navigator.xr) {
    navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        desktopInterface.style.display = 'none';
        vrContent.style.display = 'block';
      } else {
        desktopInterface.style.display = 'block';
        vrContent.style.display = 'none';
      }
    }).catch(() => {
      desktopInterface.style.display = 'block';
      vrContent.style.display = 'none';
    });
  } else {
    desktopInterface.style.display = 'block';
    vrContent.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  updateUIForDevice();
});

window.addEventListener('resize', updateUIForDevice);
