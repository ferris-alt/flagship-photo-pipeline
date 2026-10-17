// Run in a Chrome tab on https://www.flagshipcarsandclassics.com/inventory/ (the CDN and site only answer to this origin).
// Step A: collect the car links once.
//   window.fsLinks = [...new Set([...document.querySelectorAll('.vehicle-container a[href*="/inventory/"]')]
//                      .map(a => a.href.split('?')[0]).filter(h => h.split('/').length >= 8))];
//   fsLinks.length   // expect 37
//
// Step B: one car per javascript_tool call (keeps each result small enough not to be truncated):
//   await fsCar(0) ... await fsCar(36)   -> paste each JSON line into manifest.json
window.fsCar = async function (i) {
  const h = fsLinks[i];
  let html = "";
  for (let a = 0; a < 4 && html.length < 1000; a++) {
    html = await (await fetch(h, { cache: "no-store" })).text();
    if (html.length < 1000) await new Promise(r => setTimeout(r, 1500));
  }
  // cut off the "similar vehicles" block so other cars' photos don't leak in
  const cut = html.search(/SIMILAR VEHICLES|similar-vehicles|Similar Vehicles/);
  const body = cut > 0 ? html.slice(0, cut) : html;
  const ids = [...new Set(body.match(/\d{6}-[0-9a-f]{32}/g) || [])];
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script,style,noscript").forEach(s => s.remove());
  const t = (doc.body.textContent || "").replace(/\s+/g, " ");
  const stock = (t.match(/Stock Number (\S+)/) || [])[1] || h.split("/").slice(-2, -1)[0].toUpperCase();
  const title = (doc.querySelector("h1")?.textContent || "").trim().replace(/\s+/g, " ");
  const color = (t.match(/Exterior Color (\S+(?: \S+)?) Interior/) || [])[1] || "";
  const rec = { title, path: new URL(h).pathname, color, lead: ids[0], ids };
  return JSON.stringify({ [stock]: rec });
};
"ready";
