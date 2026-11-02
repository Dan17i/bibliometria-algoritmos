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
  let datos = null;
  try {
    datos = await respuesta.json();
  } catch {
    throw new Error("El servidor no pudo responder. Reinícialo si lo acabas de actualizar.");
  }
  if (!respuesta.ok) throw new Error(datos.error || "No se pudo completar la consulta.");
  return datos;
}

/** Los títulos y abstracts traen caracteres como "&": se muestran como texto, no como HTML. */
function escapar(valor) {
  const nodo = document.createElement("span");
  nodo.textContent = valor;
  return nodo.innerHTML;
}

function mostrarError(contenedor, mensaje) {
  contenedor.innerHTML = `<p class="error">${mensaje}</p>`;
}

/**
 * Intensidad del color de la matriz: más oscuro = más similar.
 *
 * Se normaliza al rango REAL de la matriz, no al teórico 0–1. Dos abstracts
 * distintos comparten pocas palabras, así que Jaccard devuelve valores entre
 * 0,10 y 0,16: mapeados sobre 0–1 todas las celdas saldrían del mismo tono y
 * la matriz no mostraría ninguna estructura.
 */
function tonoSimilitud(valor, minimo, maximo) {
  const rango = maximo - minimo;
  const intensidad = rango > 1e-9
    ? Math.max(0, Math.min(1, (valor - minimo) / rango))
    : 0.5;
  return {
    fondo: `rgba(var(--acento-rgb), ${(0.08 + intensidad * 0.87).toFixed(3)})`,
    texto: intensidad > 0.6 ? "#ffffff" : "var(--tinta)",
  };
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
          <span class="articulo__titulo">${escapar(a.titulo)}
            <small class="articulo__autores">${escapar(a.autores)}</small>
          </span>
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
  const porcentaje = Math.max(0, Math.min(1, m.similitud)) * 100;
  return `<tr>
    <td>${m.algoritmo}<div class="detalle">${m.detalle}</div></td>
    <td class="familia">${m.familia}<div class="detalle">${m.complejidad}</div></td>
    <td class="num">
      <span class="valor">${m.similitud.toFixed(4)}</span>
      <span class="barra-pista"><span class="barra" style="width:${porcentaje}%"></span></span>
    </td>
    <td class="num">
      <button type="button" class="boton-suave" data-demostrar="${m.clave}"
              data-nombre="${m.algoritmo}" data-familia="${m.familia}">Ver demostración</button>
    </td>
  </tr>`;
}

/** Los abstracts comparados, con su título y sus autores. */
function bloqueAbstracts(articulos) {
  const fichas = articulos.map((a) => `
    <details class="abstract">
      <summary>${a.id} · ${escapar(a.titulo)}</summary>
      <p class="abstract__autores">${escapar(a.autores)}</p>
      <p>${escapar(a.texto)}</p>
    </details>`).join("");
  return `<div class="abstracts"><h3>Abstracts comparados</h3>${fichas}</div>`;
}

function dibujarPar(d, contenedor) {
  const [a, b] = d.articulos;
  contenedor.innerHTML = `
    <p class="par-titulo">${a.id} · ${b.id}. Cada demostración se calcula solo cuando se abre.</p>
    <div class="tarjeta-tabla" data-par="${d.ids.join(",")}">
      <table class="medidas">
        <thead>
          <tr><th>Algoritmo</th><th>Enfoque</th><th class="num">Similitud</th><th></th></tr>
        </thead>
        <tbody>${d.algoritmos.map(filaMedida).join("")}</tbody>
      </table>
    </div>
    ${bloqueAbstracts(d.articulos)}`;
}

/** Una fila por par de artículos y una columna por algoritmo. */
function tablaDePares(d) {
  const encabezado = d.algoritmos.map((a) => `<th class="num">${a.algoritmo}</th>`).join("");
  const filas = [];
  for (let i = 0; i < d.ids.length; i++) {
    for (let j = i + 1; j < d.ids.length; j++) {
      const celdas = d.algoritmos.map((a) =>
        `<td class="num">${d.matrices[a.clave][i][j].toFixed(4)}</td>`).join("");
      filas.push(`<tr><td class="num">${d.ids[i]} · ${d.ids[j]}</td>${celdas}</tr>`);
    }
  }
  return `
    <h3 class="subtitulo-resultados">Todos los pares</h3>
    <div class="tabla-pares">
      <table class="medidas">
        <thead><tr><th>Par</th>${encabezado}</tr></thead>
        <tbody>${filas.join("")}</tbody>
      </table>
    </div>`;
}

function dibujarMatriz(d, contenedor) {
  contenedor.innerHTML = `
    <p class="par-titulo">${d.ids.length} artículos · ${d.algoritmos.length} algoritmos · ${d.ms_total.toFixed(0)} ms</p>
    <div class="selector-repr" id="selector-matriz">
      ${d.algoritmos.map((a, i) => `
        <button type="button" class="repr" data-i="${i}" aria-pressed="${i === 0}">
          ${a.algoritmo}
          <small>${a.familia}</small>
        </button>`).join("")}
    </div>
    <div id="matriz-actual"></div>
    ${tablaDePares(d)}
    ${bloqueAbstracts(d.articulos)}`;

  const selector = $("#selector-matriz");
  selector.addEventListener("click", (evento) => {
    const boton = evento.target.closest(".repr");
    if (!boton) return;
    selector.querySelectorAll(".repr").forEach((b) =>
      b.setAttribute("aria-pressed", b === boton));
    dibujarMatrizDe(d, d.algoritmos[Number(boton.dataset.i)]);
  });

  dibujarMatrizDe(d, d.algoritmos[0]);
}

function dibujarMatrizDe(d, algoritmo) {
  const matriz = d.matrices[algoritmo.clave];
  const { minimo, maximo } = rangoMatriz(matriz);
  const encabezado = d.ids.map((id) => `<th>${id}</th>`).join("");

  const filas = matriz.map((fila, i) => {
    const celdas = fila.map((v, j) => {
      if (i === j) return `<td class="diagonal">—</td>`;
      const { fondo, texto: color } = tonoSimilitud(v, minimo, maximo);
      return `<td><button type="button" class="matriz__celda" style="background:${fondo};color:${color}"
                  data-demostrar="${algoritmo.clave}" data-nombre="${algoritmo.algoritmo}"
                  data-familia="${algoritmo.familia}" data-ids="${d.ids[i]},${d.ids[j]}"
                  title="${d.ids[i]} y ${d.ids[j]}: ${v.toFixed(4)}. Ver la demostración"
                  aria-label="${d.ids[i]} y ${d.ids[j]}: ${v.toFixed(4)}. Ver la demostración">${v.toFixed(2)}</button></td>`;
    }).join("");
    return `<tr><th>${d.ids[i]}</th>${celdas}</tr>`;
  }).join("");

  $("#matriz-actual").innerHTML = `
    <div class="matriz-envoltura">
      <table class="matriz">
        <thead><tr><th></th>${encabezado}</tr></thead>
        <tbody>${filas}</tbody>
      </table>
      <p class="escala">
        <span>menos similar</span>
        <span class="escala__barra"
              style="background:linear-gradient(90deg, rgba(var(--acento-rgb),.08), rgba(var(--acento-rgb),.95))"></span>
        <span>más similar</span>
        <span class="escala__rango">${minimo.toFixed(3)} – ${maximo.toFixed(3)}</span>
      </p>
    </div>
    <p class="nota-matriz">
      Toca una casilla para ver la demostración de ese par.
      ${algoritmo.algoritmo}: ${algoritmo.complejidad} por par,
      ${algoritmo.origen === "en vivo"
        ? "calculado en el momento."
        : "leído de los 190 pares calculados una vez, porque cada par tarda varios segundos en el servidor."}
      El color se escala al rango real de esta selección, no a 0–1: los valores
      suelen caer en una franja estrecha y una escala absoluta no mostraría diferencias.
    </p>`;
}

async function calcularSimilitud() {
  const contenedor = $("#resultado-similitud");
  const boton = $("#calcular");
  contenedor.innerHTML = `<p class="cargando">Comparando…</p>`;
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

  const botones = enlaces.map((e) => `
    <button type="button" class="boton-suave" data-agrupar="${e}">Ver demostración de ${e[0].toUpperCase() + e.slice(1)}</button>`).join(" ");

  $("#dendro").innerHTML = `
    <div class="dendro__cpcc">${cpcc}</div>
    <p class="acciones-dendro">${botones}</p>
    <figure class="figura">
      <img src="${r.comparacion}" alt="Comparación de los tres criterios de enlace" loading="lazy">
      <figcaption>Los tres enlaces en la misma escala. Cada panel rotula sus propias hojas:
        cada criterio ordena los artículos de forma distinta.</figcaption>
    </figure>
    ${figuras}`;

  $("#dendro").querySelectorAll("[data-agrupar]").forEach((boton) => {
    boton.addEventListener("click", () => {
      const e = boton.dataset.agrupar;
      const nombre = `Enlace ${e[0].toUpperCase() + e.slice(1)}`;
      Demostracion.mostrarAgrupamiento(r.clave, boton.dataset.agrupar, nombre, `Agrupamiento jerárquico · ${r.etiqueta}`, boton);
    });
  });
}

/* --------------------------------------------------------------------- */
/* Requerimiento 3: búsqueda de patrones                                 */
/* --------------------------------------------------------------------- */

function resaltar(c) {
  return `<p class="contexto">${escapar(c.antes)}<mark>${escapar(c.coincidencia)}</mark>${escapar(c.despues)}</p>`;
}

let ultimoPatron = "";

async function buscarPatron(patron) {
  ultimoPatron = patron;
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
        <td class="num"><button type="button" class="boton-suave" data-patron-demo="${a.clave}"
            data-nombre="${a.algoritmo}">Ver demostración</button></td>
      </tr>`).join("");

    const hallazgos = d.por_articulo.map((a) => `
      <article class="hallazgo">
        <div class="hallazgo__cabecera">
          <span class="hallazgo__titulo">${escapar(a.titulo)}</span>
          <span class="hallazgo__id">${a.id} · ${a.ocurrencias}</span>
        </div>
        ${a.contextos.map(resaltar).join("")}
      </article>`).join("");

    contenedor.innerHTML = `
      <p class="veredicto ${d.coinciden ? "" : "discrepan"}">${veredicto}</p>
      <div class="tarjeta-tabla"><table class="medidas">
        <thead>
          <tr><th>Algoritmo</th><th class="num">Ocurrencias</th>
              <th class="num">Comparaciones</th><th class="num">Tiempo</th><th></th></tr>
        </thead>
        <tbody>${filas}</tbody>
      </table></div>
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

// La demostración se calcula solo al oprimirla: un botón por algoritmo o una casilla de la matriz
$("#resultado-similitud").addEventListener("click", (evento) => {
  const boton = evento.target.closest("[data-demostrar]");
  if (!boton) return;
  const envoltura = boton.closest("[data-par]");
  const ids = (boton.dataset.ids || envoltura.dataset.par).split(",");
  Demostracion.mostrar(boton.dataset.demostrar, ids, boton.dataset.nombre, boton.dataset.familia, boton);
});

$("#resultado-patrones").addEventListener("click", (evento) => {
  const boton = evento.target.closest("[data-patron-demo]");
  if (!boton) return;
  Demostracion.mostrarPatron(boton.dataset.patronDemo, ultimoPatron, boton.dataset.nombre, boton);
});

$("#limpiar").addEventListener("click", () => {
  elegidos.clear();
  document.querySelectorAll(".articulo input").forEach((c) => (c.checked = false));
  actualizarContador();
  $("#resultado-similitud").innerHTML =
    `<p class="vacio">Elige al menos dos artículos de la lista. La comparación se hace sobre sus abstracts completos.</p>`;
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
