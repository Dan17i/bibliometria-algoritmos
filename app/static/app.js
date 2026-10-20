/* =========================================================================
   Interacción de la aplicación
   -------------------------------------------------------------------------
   Todo el cálculo ocurre en el servidor; aquí solo se recogen las opciones,
   se pide el resultado y se dibuja. Sin dependencias externas.
   ========================================================================= */

const elegidos = new Set();

/* --------------------------------------------------------------------- */
/* Utilidades                                                            */
/* --------------------------------------------------------------------- */

const $ = (sel) => document.querySelector(sel);

function texto(elemento, valor) {
  elemento.textContent = valor;
}

/**
 * Separador de miles con espacio fino, no con punto.
 *
 * En la misma tabla conviven enteros grandes ("29 619 comparaciones") y
 * decimales ("2.772 ms"). Con el punto como separador de miles, "29.619"
 * se lee como un decimal y la tabla se vuelve ambigua. El espacio fino es
 * la convención SI y elimina la confusión.
 */
function miles(n) {
  return n.toLocaleString("es-CO").replace(/\./g, " ");
}

async function pedir(ruta, opciones) {
  const respuesta = await fetch(ruta, opciones);
  const datos = await respuesta.json();
  if (!respuesta.ok) throw new Error(datos.error || "No se pudo completar la consulta.");
  return datos;
}

function mostrarError(contenedor, mensaje) {
  contenedor.innerHTML = `<p class="error">${mensaje}</p>`;
}

/**
 * Escala de azules para la matriz: más oscuro = más similar.
 *
 * La intensidad se normaliza al rango REAL de la matriz, no al teórico 0–1.
 * Dos abstracts distintos comparten pocas palabras, así que Jaccard devuelve
 * valores entre 0,10 y 0,16: mapeados sobre 0–1 todas las celdas saldrían
 * del mismo tono y la matriz no mostraría ninguna estructura.
 */
function tonoSimilitud(valor, minimo, maximo) {
  const rango = maximo - minimo;
  const intensidad = rango > 1e-9
    ? Math.max(0, Math.min(1, (valor - minimo) / rango))
    : 0.5;
  // Se mezcla el azul contra el papel en vez de usar opacidad: así el texto
  // encima conserva su contraste.
  const r = Math.round(247 - (247 - 31) * intensidad);
  const g = Math.round(248 - (248 - 78) * intensidad);
  const b = Math.round(250 - (250 - 121) * intensidad);
  return { fondo: `rgb(${r},${g},${b})`, texto: intensidad > 0.6 ? "#ffffff" : "#16202e" };
}

/** Mínimo y máximo de la matriz, ignorando la diagonal (siempre 1). */
function rangoMatriz(matriz) {
  const fuera = [];
  matriz.forEach((fila, i) =>
    fila.forEach((v, j) => { if (i !== j) fuera.push(v); }));
  return { minimo: Math.min(...fuera), maximo: Math.max(...fuera) };
}

/* --------------------------------------------------------------------- */
/* Ficha del corpus                                                      */
/* --------------------------------------------------------------------- */

async function cargarFicha() {
  try {
    const d = await pedir("/api/corpus");
    texto($("#d-articulos"), d.articulos);
    texto($("#d-palabras"), miles(d.palabras));
    texto($("#d-caracteres"), miles(d.caracteres));
  } catch {
    /* La ficha es informativa: si falla, el resto de la página sigue útil. */
  }
}

/* --------------------------------------------------------------------- */
/* Requerimiento 1: similitud                                            */
/* --------------------------------------------------------------------- */

async function cargarArticulos() {
  const lista = $("#lista-articulos");
  try {
    const d = await pedir("/api/articulos");
    lista.innerHTML = d.articulos.map((a) => `
      <li>
        <label class="articulo">
          <input type="checkbox" value="${a.id}">
          <span class="articulo__id">${a.id}</span>
          <span class="articulo__titulo">${a.titulo}</span>
        </label>
      </li>`).join("");

    lista.addEventListener("change", (evento) => {
      const casilla = evento.target;
      if (casilla.checked) elegidos.add(casilla.value);
      else elegidos.delete(casilla.value);
      actualizarContador();
    });
  } catch (error) {
    lista.innerHTML = `<li><p class="error">${error.message}</p></li>`;
  }
}

function actualizarContador() {
  texto($("#n-elegidos"), elegidos.size);
  $("#calcular").disabled = elegidos.size < 2;
}

function filaMedida(m) {
  if (m.similitud === null) {
    return `<tr>
      <td>${m.algoritmo}<div class="detalle">${m.detalle}</div></td>
      <td class="familia">${m.familia}</td>
      <td class="num ausente" colspan="2">sin medir</td>
    </tr>`;
  }
  const porcentaje = Math.max(0, Math.min(1, m.similitud)) * 100;
  return `<tr>
    <td>${m.algoritmo}<div class="detalle">${m.detalle}</div></td>
    <td class="familia">${m.familia}</td>
    <td class="num">
      <span class="valor">${m.similitud.toFixed(4)}</span>
      <span class="barra" style="width:${porcentaje}%"></span>
    </td>
    <td class="num">${m.ms.toFixed(2)} ms<div class="detalle">${m.complejidad}</div></td>
  </tr>`;
}

function dibujarPar(d, contenedor) {
  const [a, b] = d.articulos;
  contenedor.innerHTML = `
    <p class="par-titulo">${a.id} · ${b.id} — calculado en ${d.ms_total.toFixed(0)} ms</p>
    <table class="medidas">
      <thead>
        <tr><th>Algoritmo</th><th>Enfoque</th><th class="num">Similitud</th><th class="num">Tiempo</th></tr>
      </thead>
      <tbody>${d.algoritmos.map(filaMedida).join("")}</tbody>
    </table>`;
}

function dibujarMatriz(d, contenedor) {
  const { minimo, maximo } = rangoMatriz(d.matriz);
  const encabezado = d.ids.map((id) => `<th>${id}</th>`).join("");

  const filas = d.matriz.map((fila, i) => {
    const celdas = fila.map((v, j) => {
      if (i === j) return `<td class="diagonal" style="background:var(--papel-hondo)">—</td>`;
      const { fondo, texto: color } = tonoSimilitud(v, minimo, maximo);
      return `<td style="background:${fondo};color:${color}"
                  title="${d.ids[i]} vs ${d.ids[j]}: ${v.toFixed(4)}">${v.toFixed(2)}</td>`;
    }).join("");
    return `<tr><th>${d.ids[i]}</th>${celdas}</tr>`;
  }).join("");

  contenedor.innerHTML = `
    <p class="par-titulo">${d.ids.length} artículos · ${d.metrica} · ${d.ms_total.toFixed(0)} ms</p>
    <div class="matriz-envoltura">
      <table class="matriz">
        <thead><tr><th></th>${encabezado}</tr></thead>
        <tbody>${filas}</tbody>
      </table>
      <p class="escala">
        <span>menos similar</span>
        <span class="escala__barra"
              style="background:linear-gradient(90deg, rgb(247,248,250), rgb(31,78,121))"></span>
        <span>más similar</span>
        <span class="escala__rango">${minimo.toFixed(3)} – ${maximo.toFixed(3)}</span>
      </p>
    </div>
    <p class="detalle" style="margin-top:1rem;color:var(--tinta-tenue);font-size:.87rem;max-width:68ch">
      El color se escala al rango real de esta selección, no a 0–1: dos abstracts
      distintos comparten pocas palabras, así que todos los valores caen en una
      franja estrecha y una escala absoluta no mostraría diferencias.
      Con más de dos artículos se usa una medida rápida, porque los algoritmos de
      programación dinámica sobre todos los pares tardarían demasiado en vivo.
    </p>`;
}

async function calcularSimilitud() {
  const contenedor = $("#resultado-similitud");
  const boton = $("#calcular");
  contenedor.innerHTML = `<p class="cargando">Calculando sobre los abstracts completos…</p>`;
  boton.disabled = true;

  try {
    const d = await pedir("/api/similitud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [...elegidos] }),
    });
    if (d.modo === "par") dibujarPar(d, contenedor);
    else dibujarMatriz(d, contenedor);
  } catch (error) {
    mostrarError(contenedor, error.message);
  } finally {
    boton.disabled = elegidos.size < 2;
  }
}

/* --------------------------------------------------------------------- */
/* Requerimiento 2: dendrogramas                                         */
/* --------------------------------------------------------------------- */

let representaciones = [];

async function cargarClustering() {
  const selector = $("#selector-repr");
  try {
    const d = await pedir("/api/clustering");
    representaciones = d.representaciones;
    if (!representaciones.length) {
      mostrarError($("#dendro"), "No hay dendrogramas generados todavía. Ejecuta python -m python_viz.dendrograma");
      return;
    }

    selector.innerHTML = representaciones.map((r, i) => `
      <button type="button" class="repr" data-i="${i}" aria-pressed="${i === 0}">
        ${r.etiqueta}
        <small>CPCC ${r.cpcc[r.mejor_enlace].toFixed(3)} · ${r.mejor_enlace}</small>
      </button>`).join("");

    selector.addEventListener("click", (evento) => {
      const boton = evento.target.closest(".repr");
      if (!boton) return;
      selector.querySelectorAll(".repr").forEach((b) =>
        b.setAttribute("aria-pressed", b === boton));
      dibujarDendro(Number(boton.dataset.i));
    });

    dibujarDendro(0);
  } catch (error) {
    mostrarError($("#dendro"), error.message);
  }
}

function dibujarDendro(indice) {
  const r = representaciones[indice];
  const enlaces = Object.keys(r.cpcc);

  const cpcc = enlaces.map((e) => `
    <div class="${e === r.mejor_enlace ? "mejor" : ""}">
      <span>${e}</span><span>${r.cpcc[e].toFixed(4)}</span>
    </div>`).join("");

  const figuras = enlaces.map((e) => `
    <figure class="figura">
      <img src="${r.figuras[e]}" alt="Dendrograma con enlace ${e} usando ${r.etiqueta}" loading="lazy">
      <figcaption>Enlace ${e}${e === r.mejor_enlace ? " — el más fiel a las distancias originales" : ""}</figcaption>
    </figure>`).join("");

  $("#dendro").innerHTML = `
    <div class="dendro__cpcc">${cpcc}</div>
    <figure class="figura">
      <img src="${r.comparacion}" alt="Comparación de los tres criterios de enlace" loading="lazy">
      <figcaption>Los tres enlaces en la misma escala. Cada panel rotula sus propias hojas:
        cada criterio ordena los artículos de forma distinta.</figcaption>
    </figure>
    ${figuras}`;
}

/* --------------------------------------------------------------------- */
/* Requerimiento 3: búsqueda de patrones                                 */
/* --------------------------------------------------------------------- */

function resaltar(c) {
  return `<p class="contexto">${c.antes}<mark>${c.coincidencia}</mark>${c.despues}</p>`;
}

async function buscarPatron(patron) {
  const contenedor = $("#resultado-patrones");
  contenedor.innerHTML = `<p class="cargando">Recorriendo los veinte abstracts…</p>`;

  try {
    const d = await pedir("/api/buscar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patron }),
    });

    const veredicto = d.coinciden
      ? `Los tres algoritmos encontraron las mismas ${d.total} ocurrencias en ${d.articulos_con_coincidencia} artículos.`
      : `Los algoritmos no coinciden entre sí. Hay un error en alguna implementación.`;

    const filas = d.algoritmos.map((a) => `
      <tr>
        <td>${a.algoritmo}<div class="detalle">${a.extra}</div></td>
        <td class="num">${a.ocurrencias}</td>
        <td class="num">${miles(a.comparaciones)}</td>
        <td class="num"><span class="valor">${a.ms.toFixed(3)} ms</span></td>
      </tr>`).join("");

    const hallazgos = d.por_articulo.map((a) => `
      <article class="hallazgo">
        <div class="hallazgo__cabecera">
          <span class="hallazgo__titulo">${a.titulo}</span>
          <span class="hallazgo__id">${a.id} · ${a.ocurrencias}</span>
        </div>
        ${a.contextos.map(resaltar).join("")}
      </article>`).join("");

    contenedor.innerHTML = `
      <p class="veredicto ${d.coinciden ? "" : "discrepan"}">${veredicto}</p>
      <table class="medidas">
        <thead>
          <tr><th>Algoritmo</th><th class="num">Ocurrencias</th>
              <th class="num">Comparaciones</th><th class="num">Tiempo</th></tr>
        </thead>
        <tbody>${filas}</tbody>
      </table>
      ${d.por_articulo.length ? `
        <div class="coincidencias">
          <h3>Dónde aparece</h3>
          ${hallazgos}
        </div>` : `<p class="vacio" style="margin-top:1.5rem">
          El patrón no aparece en ningún abstract del corpus.</p>`}`;
  } catch (error) {
    mostrarError(contenedor, error.message);
  }
}

/* --------------------------------------------------------------------- */
/* Arranque                                                              */
/* --------------------------------------------------------------------- */

$("#calcular").addEventListener("click", calcularSimilitud);

$("#limpiar").addEventListener("click", () => {
  elegidos.clear();
  document.querySelectorAll(".articulo input").forEach((c) => (c.checked = false));
  actualizarContador();
  $("#resultado-similitud").innerHTML =
    `<p class="vacio">Elige al menos dos artículos de la lista. El cálculo ocurre en el momento, sobre los abstracts completos.</p>`;
});

$("#form-buscar").addEventListener("submit", (evento) => {
  evento.preventDefault();
  buscarPatron($("#patron").value);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#patron").value = chip.dataset.patron;
    buscarPatron(chip.dataset.patron);
  });
});

cargarFicha();
cargarArticulos();
cargarClustering();
buscarPatron($("#patron").value);
