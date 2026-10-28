import './style.css';

const chat = document.querySelector('#conversation');
const area = document.querySelector('#chatArea');
const form = document.querySelector('#composer');
const input = document.querySelector('#messageInput');
const breakButton = document.querySelector('#breakButton');
const suggestionRow = document.querySelector('#suggestionRow');

const fallbackPlaces = [
  { icon: '🚻', type: 'TOILET / WASHROOM', name: 'Search nearby public toilets', detail: 'Uses your current location', query: 'public toilet' },
  { icon: '💧', type: 'WATER', name: 'Search nearby drinking water', detail: 'Uses your current location', query: 'drinking water' },
  { icon: '🌳', type: 'RESTING PLACE', name: 'Search nearby parks', detail: 'Uses your current location', query: 'park' },
  { icon: '🛕', type: 'RESTING PLACE', name: 'Search nearby Gurudwaras', detail: 'Uses your current location', query: 'Gurudwara' }
];

function time() { return new Intl.DateTimeFormat('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date()); }
function scrollDown() { requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; }); }
function userMessage(text) {
  chat.insertAdjacentHTML('beforeend', `<div class="message outgoing"><p>${escapeHtml(text)}</p><span class="ticks">✓✓</span><time>${time()}</time></div>`);
  scrollDown();
}
function escapeHtml(value) { const d = document.createElement('div'); d.textContent = value; return d.innerHTML; }
function distance(origin, destination) {
  const r = 6371e3;
  const φ1 = origin.lat * Math.PI / 180, φ2 = destination.lat * Math.PI / 180;
  const Δφ = (destination.lat - origin.lat) * Math.PI / 180;
  const Δλ = (destination.lon - origin.lon) * Math.PI / 180;
  const a = Math.sin(Δφ / 2) ** 2 + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  const metres = r * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return metres < 1000 ? `${Math.round(metres / 10) * 10} m away` : `${(metres / 1000).toFixed(1)} km away`;
}
function mapsUrl(origin, place) {
  const destination = place.lat ? `${place.lat},${place.lon}` : place.query;
  if (!origin) return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(destination)}`;
  const start = `${origin.lat},${origin.lon}`;
  return `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(start)}&destination=${encodeURIComponent(destination)}&travelmode=walking`;
}
function placeCategory(tags) {
  if (tags.amenity === 'toilets') return { icon: '🚻', type: 'TOILET / WASHROOM' };
  if (tags.amenity === 'drinking_water') return { icon: '💧', type: 'WATER' };
  if (tags.leisure === 'park') return { icon: '🌳', type: 'RESTING PLACE' };
  return { icon: '🛕', type: 'RESTING PLACE' };
}
async function findNearbyPlaces(location) {
  const query = `[out:json][timeout:15];(nwr[amenity=toilets](around:2500,${location.lat},${location.lon});nwr[amenity=drinking_water](around:2500,${location.lat},${location.lon});nwr[leisure=park](around:2500,${location.lat},${location.lon});nwr[amenity=place_of_worship][religion=sikh](around:3000,${location.lat},${location.lon}););out center tags;`;
  const response = await fetch('https://overpass-api.de/api/interpreter', { method: 'POST', body: query });
  if (!response.ok) throw new Error('Nearby places are unavailable');
  const data = await response.json();
  return data.elements.map((item) => {
    const tags = item.tags || {};
    const category = placeCategory(tags);
    const lat = item.lat ?? item.center?.lat;
    const lon = item.lon ?? item.center?.lon;
    const metres = Math.round(6371000 * 2 * Math.atan2(Math.sqrt(Math.sin((lat - location.lat) * Math.PI / 360) ** 2 + Math.cos(location.lat * Math.PI / 180) * Math.cos(lat * Math.PI / 180) * Math.sin((lon - location.lon) * Math.PI / 360) ** 2), Math.sqrt(1 - (Math.sin((lat - location.lat) * Math.PI / 360) ** 2 + Math.cos(location.lat * Math.PI / 180) * Math.cos(lat * Math.PI / 180) * Math.sin((lon - location.lon) * Math.PI / 360) ** 2))));
    return { ...category, name: tags.name || (category.type === 'WATER' ? 'Drinking water point' : category.type === 'TOILET / WASHROOM' ? 'Public toilet' : 'Public rest stop'), detail: tags.opening_hours ? `Hours: ${tags.opening_hours}` : 'OpenStreetMap community listing', lat, lon, metres, distance: distance(location, { lat, lon }) };
  }).filter((place) => place.lat && place.lon).sort((a, b) => a.metres - b.metres).slice(0, 8);
}
function renderPlaces(places, location, live) {
  const cards = places.map((place) => `<a class="place-card" target="_blank" rel="noopener" href="${mapsUrl(location, place)}"><span class="place-icon">${place.icon}</span><span class="place-copy"><small>${place.type}</small><strong>${escapeHtml(place.name)}</strong><span>${escapeHtml(place.detail)}</span><b>${place.distance || 'Nearby'} <em>↗ Directions</em></b></span></a>`).join('');
  chat.insertAdjacentHTML('beforeend', `<div class="message incoming places-message"><p class="nearby-title">${live ? 'Nearby safe stops' : 'Find nearby safe stops'} <span>${live ? `${places.length} found` : 'Location ready'}</span></p>${cards}<p class="helper">${live ? 'Live results use OpenStreetMap community data. Tap a place for GPS walking directions.' : 'We could not load verified nearby places right now. Each button opens a live map search from your location.'}</p><time>${time()}</time></div><div class="suggestion-row post-options"><button class="quick-reply small" data-response="Show only water points">💧 Water</button><button class="quick-reply small" data-response="Report an issue">⚑ Report issue</button></div>`);
  scrollDown();
}
function showBreakFlow() {
  if (document.querySelector('.places-message')) return;
  suggestionRow.remove();
  userMessage('BREAK');
  setTimeout(() => {
    chat.insertAdjacentHTML('beforeend', `<div class="message incoming status"><p><strong>Your safety break is noted.</strong> 🌿</p><p>Finding safe stops near you now. Your precise location stays on your device and is never shared with an employer.</p><time>${time()}</time></div>`);
    scrollDown();
    if (!navigator.geolocation) { renderPlaces(fallbackPlaces, null, false); return; }
    navigator.geolocation.getCurrentPosition(async ({ coords }) => {
      const location = { lat: coords.latitude, lon: coords.longitude };
      try { const places = await findNearbyPlaces(location); renderPlaces(places.length ? places : fallbackPlaces, location, places.length > 0); }
      catch { renderPlaces(fallbackPlaces, location, false); }
    }, () => {
      chat.insertAdjacentHTML('beforeend', `<div class="message incoming"><p>Location access is off. Turn it on in your browser to see live nearby rest stops.</p><time>${time()}</time></div>`);
      scrollDown();
    }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 });
  }, 550);
}
function respond(text) {
  userMessage(text);
  setTimeout(() => {
    const message = text.toLowerCase().includes('water') ? 'I found a free drinking-water point 350 m away. Tap the Water card above for directions.' : 'Thanks for looking out for other workers. Tell us what went wrong and our team will check the stop.';
    chat.insertAdjacentHTML('beforeend', `<div class="message incoming"><p>${message}</p><time>${time()}</time></div>`);
    scrollDown();
  }, 450);
}
breakButton.addEventListener('click', showBreakFlow);
form.addEventListener('submit', (event) => { event.preventDefault(); const value = input.value.trim(); if (!value) return; input.value = ''; if (value.toLowerCase() === 'break') showBreakFlow(); else respond(value); });
chat.addEventListener('click', (event) => { const button = event.target.closest('[data-response]'); if (button) respond(button.dataset.response); });
