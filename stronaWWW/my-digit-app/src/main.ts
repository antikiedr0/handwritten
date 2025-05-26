const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const predictionDiv = document.getElementById('prediction')!;

let drawing = false;

// Bufor obrazu: 28x28, ale wizualnie skalujemy
canvas.width = 28;
canvas.height = 28;
canvas.style.width = '280px';
canvas.style.height = '280px';

// Białe tło
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Styl rysowania – czarna linia, gruba
ctx.strokeStyle = 'black';
ctx.lineWidth = 1;  // tylko 1px, bo rysujemy bezpośrednio w 28x28
ctx.lineCap = 'round';

function getMousePos(e: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const pos = getMousePos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const pos = getMousePos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
  sendImage();
});

function sendImage() {
  const imageData = canvas.toDataURL('image/png');

  fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  })
    .then((res) => res.json())
    .then((data) => {
      predictionDiv.textContent = `Rozpoznana cyfra: ${data.prediction}`;
    })
    .catch((err) => {
      predictionDiv.textContent = 'Błąd podczas predykcji';
      console.error(err);
    });

  // Wyczyść canvas
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
}
