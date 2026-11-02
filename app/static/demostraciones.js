/* =========================================================================
   Demostración matemática de cada algoritmo
   -------------------------------------------------------------------------
   Todos los números vienen del servidor, que ejecuta solo el algoritmo que se
   oprimió. Aquí únicamente se dibujan y se escriben las fórmulas con KaTeX.
   ========================================================================= */

const Demostracion = (() => {
  const hoja = document.querySelector("#hoja");
  const cuerpo = document.querySelector("#hoja-cuerpo");
  const titulo = document.querySelector("#hoja-titulo");
  const familia = document.querySelector("#hoja-familia");
  let solicitud = null;     // para cancelar la petición si se cierra antes de que llegue
  let retorno = null;       // cómo volver a la demostración original desde "tus palabras"

  const seguro = (valor) => {
    const nodo = document.createElement("span");
    nodo.textContent = valor;
    return nodo.innerHTML;
  };

  const formula = (latex) =>
    `<div class="dem-formula">${katex.renderToString(latex, { displayMode: true, throwOnError: false })}</div>`;

  const caracter = (c) => (c === " " ? "␣" : seguro(c));
  const nombreCaracter = (c) => (c === " " ? "espacio" : `«${c}»`);

  /* ------------------------------------------------------------------- */
  /* Abrir y cerrar: sale del botón que la abrió y vuelve a él             */
  /* ------------------------------------------------------------------- */

  function abrir(disparador) {
    if (!hoja.open) hoja.showModal();
    if (disparador) {
      const caja = disparador.getBoundingClientRect();
      const propia = hoja.getBoundingClientRect();
      hoja.style.transformOrigin =
        `${caja.left + caja.width / 2 - propia.left}px ${caja.top + caja.height / 2 - propia.top}px`;
    }
    // Un cuadro después, para que la transición parta del estado cerrado
    requestAnimationFrame(() => hoja.classList.add("abierta"));
    cuerpo.scrollTop = 0;
  }

  function cerrar() {
    if (solicitud) solicitud.abort();
    hoja.classList.remove("abierta");
    const alTerminar = () => {
      // Si se volvió a abrir mientras se cerraba, no se interrumpe
      if (!hoja.classList.contains("abierta") && hoja.open) hoja.close();
    };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setTimeout(alTerminar, 200);
    } else {
      setTimeout(alTerminar, 320);
    }
  }

  document.querySelector("#hoja-cerrar").addEventListener("click", cerrar);
  hoja.addEventListener("cancel", (evento) => { evento.preventDefault(); cerrar(); });
  hoja.addEventListener("click", (evento) => { if (evento.target === hoja) cerrar(); });

  /* ------------------------------------------------------------------- */
  /* Peticiones                                                            */
  /* ------------------------------------------------------------------- */

  async function pedir(cuerpoPeticion, ruta = "/api/demostracion") {
    if (solicitud) solicitud.abort();
    solicitud = new AbortController();
    const respuesta = await fetch(ruta, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cuerpoPeticion),
      signal: solicitud.signal,
    });
    let datos = null;
    try {
      datos = await respuesta.json();
    } catch {
      // El servidor respondió con una página de error y no con datos
      throw new Error(respuesta.status === 404
        ? "El servidor no tiene esta función. Reinícialo para cargar la versión nueva."
        : "El servidor no pudo calcularlo. Inténtalo de nuevo.");
    }
    if (!respuesta.ok) throw new Error(datos.error || "No se pudo calcular.");
    return datos;
  }

  async function mostrar(clave, ids, nombre, grupo, disparador) {
    titulo.textContent = nombre;
    familia.textContent = grupo;
    cuerpo.innerHTML = `<p class="hoja__espera" role="status">Ejecutando ${seguro(nombre)} sobre ${ids.join(" y ")}…</p>`;
    retorno = { etiqueta: `Volver a ${ids.join(" y ")}`, accion: () => mostrar(clave, ids, nombre, grupo, null) };
    abrir(disparador);
    try {
      dibujar(await pedir({ algoritmo: clave, ids }));
    } catch (error) {
      if (error.name !== "AbortError") {
        cuerpo.innerHTML = `<p class="error">${seguro(error.message)}</p>`;
      }
    }
  }

  /* ------------------------------------------------------------------- */
  /* Partes de la demostración                                             */
  /* ------------------------------------------------------------------- */

  function cabecera(d) {
    const quienes = d.subtitulo ? seguro(d.subtitulo) : d.textos
      ? `«${seguro(d.textos[0])}» y «${seguro(d.textos[1])}»`
      : d.articulos.map((a) => a.id).join(" y ");
    return `
      <div class="dem-resultado">
        <span class="dem-valor">${d.valor_texto ?? d.similitud.toFixed(4)}</span>${d.etiqueta_valor ? `<span class="dem-par">${d.etiqueta_valor}</span>` : ""}
        <span class="dem-par">${quienes}${d.ms ? ` · ${d.ms.toFixed(d.ms < 10 ? 2 : 0)} ms` : ""}</span>
      </div>
      <p class="dem-frase">${seguro(d.en_una_frase)}</p>
      ${d.que_significa ? `<p class="dem-guia">${seguro(d.que_significa)}</p>` : ""}`;
  }

  /** Una parte de la demostración; `guia` dice en una frase qué se va a encontrar en ella. */
  function seccion(nombre, contenido, guia = "") {
    return `<section class="dem-seccion"><h3>${nombre}</h3>${guia ? `<p class="dem-guia">${guia}</p>` : ""}${contenido}</section>`;
  }

  function pasos(lista) {
    return lista.map((p, i) => `
      <div class="dem-tarjeta dem-paso">
        <span class="dem-paso__numero">${i + 1}</span>
        <span class="dem-paso__titulo">${seguro(p.titulo)}</span>
        ${p.texto ? `<p>${seguro(p.texto)}</p>` : ""}
        ${p.latex.map(formula).join("")}
      </div>`).join("");
  }

  /* --- matriz de programación dinámica ---------------------------------- */

  function matrizHtml(v, interactiva) {
    const encabezado = (interactiva ? ["ε", ...v.columnas] : ["", ...v.columnas])
      .map((c) => `<th>${caracter(c)}</th>`).join("");
    const filas = v.celdas.map((fila, i) => {
      let rotulo = "";
      if (interactiva) rotulo = i === 0 ? "ε" : v.filas[i - 1];
      else if (i > 0) rotulo = v.filas[i - 1];
      const celdas = fila.map((valor, j) => (interactiva
        ? `<td><button type="button" class="pd__celda" data-i="${i}" data-j="${j}">${valor}</button></td>`
        : `<td><span class="pd__celda" style="display:grid;place-items:center;cursor:default">${valor}</span></td>`)).join("");
      return `<tr><th>${caracter(rotulo)}</th>${celdas}</tr>`;
    }).join("");
    return `<div class="pd-envoltura"><table class="pd"><thead><tr><th></th>${encabezado}</tr></thead><tbody>${filas}</tbody></table></div>`;
  }

  function explicarCelda(v, i, j) {
    const F = v.simbolo;
    const c = v.celdas;
    if (i === 0 && j === 0) {
      return { vecinas: [], ganadora: null, latex: `${F}(0, 0) = 0`, texto: "Dos cadenas vacías: no hay nada que cambiar." };
    }
    if (i === 0 || j === 0) {
      const k = i + j;
      const costo = v.operacion === "min" ? `${k}` : `${k} \\cdot (-1)`;
      const texto = v.operacion === "min"
        ? `Convertir ${k} carácter${k > 1 ? "es" : ""} en la cadena vacía cuesta ${k}.`
        : `Empezar con ${k} hueco${k > 1 ? "s" : ""} seguido${k > 1 ? "s" : ""}: cada uno resta 1.`;
      return { vecinas: [], ganadora: null, latex: `${F}(${i}, ${j}) = ${costo} = ${c[i][j]}`, texto };
    }

    const a = v.filas[i - 1];
    const b = v.columnas[j - 1];
    const iguales = a === b;
    const vecinas = [[i - 1, j - 1], [i - 1, j], [i, j - 1]];
    let candidatos;
    let latex;
    let texto;
    if (v.operacion === "min") {
      const costo = iguales ? 0 : 1;
      candidatos = [c[i - 1][j - 1] + costo, c[i - 1][j] + 1, c[i][j - 1] + 1];
      latex = `${F}(${i}, ${j}) = \\min(\\underbrace{${c[i - 1][j - 1]} + ${costo}}_{\\text{sustituir}},\\; \\underbrace{${c[i - 1][j]} + 1}_{\\text{eliminar}},\\; \\underbrace{${c[i][j - 1]} + 1}_{\\text{insertar}}) = ${c[i][j]}`;
      texto = `Se compara ${nombreCaracter(a)} con ${nombreCaracter(b)}: ${iguales ? "coinciden, así que pasar por la diagonal no cuesta" : "no coinciden, así que sustituir cuesta 1"}.`;
    } else {
      const s = iguales ? 1 : -1;
      candidatos = [c[i - 1][j - 1] + s, c[i - 1][j] - 1, c[i][j - 1] - 1];
      latex = `${F}(${i}, ${j}) = \\max(\\underbrace{${c[i - 1][j - 1]} + (${s})}_{\\text{alinear}},\\; \\underbrace{${c[i - 1][j]} - 1}_{\\text{hueco}},\\; \\underbrace{${c[i][j - 1]} - 1}_{\\text{hueco}}) = ${c[i][j]}`;
      texto = `Se compara ${nombreCaracter(a)} con ${nombreCaracter(b)}: ${iguales ? "coinciden, alinearlos suma 1" : "no coinciden, alinearlos resta 1"}.`;
    }
    const ganadora = candidatos.findIndex((x) => x === c[i][j]);
    return { vecinas, ganadora: vecinas[ganadora], latex, texto };
  }

  function activarMatriz(raiz, v) {
    const filas = v.celdas.length;
    const columnas = v.celdas[0].length;
    const celdas = raiz.querySelectorAll(".pd__celda");
    const rango = raiz.querySelector("input[type=range]");
    const contador = raiz.querySelector(".pd-control span");
    const explica = raiz.querySelector(".pd-explica");

    function pintar(orden) {
      const i = Math.floor(orden / columnas);
      const j = orden % columnas;
      const e = explicarCelda(v, i, j);
      celdas.forEach((celda) => {
        const a = Number(celda.dataset.i);
        const b = Number(celda.dataset.j);
        celda.className = "pd__celda";
        if (a * columnas + b > orden) celda.classList.add("futura");
        if (e.vecinas.some(([x, y]) => x === a && y === b)) celda.classList.add("vecina");
        if (e.ganadora && e.ganadora[0] === a && e.ganadora[1] === b) celda.classList.add("ganadora");
        if (a === i && b === j) celda.classList.add("actual");
      });
      explica.innerHTML = `${formula(e.latex)}<p>${seguro(e.texto)}</p>`;
      contador.textContent = `celda ${orden + 1} de ${filas * columnas}`;
    }

    rango.addEventListener("input", () => pintar(Number(rango.value)));
    raiz.querySelector(".pd").addEventListener("click", (evento) => {
      const celda = evento.target.closest(".pd__celda");
      if (!celda) return;
      const orden = Number(celda.dataset.i) * columnas + Number(celda.dataset.j);
      rango.value = orden;
      pintar(orden);
    });
    // Se abre en la primera celda que no es borde: ahí empieza lo interesante
    const inicial = Math.min(columnas + 1, filas * columnas - 1);
    rango.value = inicial;
    pintar(inicial);
  }

  function visualMatriz(v) {
    const total = v.celdas.length * v.celdas[0].length;
    const pregunta = v.operacion === "min"
      ? "¿cuántos cambios hacen falta para convertir el pedazo del primer texto que llega hasta esa fila en el pedazo del segundo que llega hasta esa columna?"
      : "¿cuál es el mejor puntaje al alinear el pedazo del primer texto que llega hasta esa fila con el pedazo del segundo que llega hasta esa columna?";
    const aviso = `Cada casilla responde: ${pregunta}
      ${v.completa ? "Aquí está la tabla completa." : "Aquí están solo las primeras casillas, porque la tabla de dos abstracts es enorme."}
      Mueve el control o toca una casilla: la azul es la que se calcula, las claras son sus vecinas y la del borde es la opción elegida. ε significa «texto vacío» y ␣ un espacio.`;
    return `
      <div class="dem-tarjeta" data-matriz>
        <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">${aviso}</p>
        ${matrizHtml(v, true)}
        <div class="pd-control">
          <input type="range" min="0" max="${total - 1}" value="0" aria-label="Avanzar por las celdas de la matriz">
          <span></span>
        </div>
        <div class="pd-explica" aria-live="polite"></div>
        ${v.final ? `
          <p class="dem-subtitulo">Las últimas casillas. La de la esquina inferior derecha es el resultado final.</p>
          ${matrizHtml(v.final, false)}` : ""}
      </div>`;
  }

  /* --- conjuntos, términos, palabras, tablas y ángulo --------------------- */

  function fichas(lista) {
    return `<div class="fichas">${lista.map((p) => `<span class="ficha-palabra">${seguro(p)}</span>`).join("")}</div>`;
  }

  function visualConjuntos(v) {
    return `
      <div class="dem-tarjeta conjuntos">
        <div class="conjunto"><h4>Solo en el primero · ${v.solo_a.length}</h4>${fichas(v.solo_a)}</div>
        <div class="conjunto conjunto--comun"><h4>En los dos · ${v.comunes.length}</h4>${fichas(v.comunes)}</div>
        <div class="conjunto"><h4>Solo en el segundo · ${v.solo_b.length}</h4>${fichas(v.solo_b)}</div>
      </div>`;
  }

  function visualTerminos(v) {
    const mayor = v.terminos.length ? Math.max(v.terminos[0].producto, 1e-9) : 1;
    const filas = v.terminos.map((t) => `
      <tr>
        <td>${seguro(t.termino)}</td>
        <td>${t.tf_a}</td><td>${t.tf_b}</td>
        <td>${t.idf.toFixed(4)}</td>
        <td>${t.peso_a.toFixed(4)}</td><td>${t.peso_b.toFixed(4)}</td>
        <td>${t.producto.toFixed(4)}</td>
        <td class="aporte"><span style="width:${Math.max(0, (t.producto / mayor) * 100)}%"></span></td>
      </tr>`).join("");
    return `
      <div class="dem-tarjeta">
        <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">Las palabras que más ayudan a que los dos textos se parezcan. tf: cuántas veces aparece en cada texto. idf: qué tan poco común es en los 20 abstracts. w: su peso final (tf × idf). La última columna es lo que aporta esa palabra.</p>
        <div style="overflow-x:auto"><table class="tabla-dem">
          <thead><tr><th>Palabra</th><th>tf A</th><th>tf B</th><th>idf</th><th>w A</th><th>w B</th><th>w A · w B</th><th></th></tr></thead>
          <tbody>${filas}</tbody>
        </table></div>
      </div>`;
  }

  function visualPalabras(v) {
    const fuera = (lista) => (lista.length ? ` No están en el diccionario de GloVe: ${seguro([...new Set(lista)].join(", "))}.` : "");
    return `
      <div class="dem-tarjeta conjuntos" style="grid-template-columns:repeat(auto-fit,minmax(14rem,1fr))">
        <div class="conjunto"><h4>Primer abstract · ${v.total_a} palabras</h4>${fichas(v.a)}<p style="font-size:.8125rem;color:var(--tinta-tenue);margin:.5rem 0 0">${v.a.length < v.total_a ? `Primeras ${v.a.length} de ${v.total_a} palabras.` : ""}${fuera(v.fuera_a)}</p></div>
        <div class="conjunto"><h4>Segundo abstract · ${v.total_b} palabras</h4>${fichas(v.b)}<p style="font-size:.8125rem;color:var(--tinta-tenue);margin:.5rem 0 0">${v.b.length < v.total_b ? `Primeras ${v.b.length} de ${v.total_b} palabras.` : ""}${fuera(v.fuera_b)}</p></div>
      </div>`;
  }

  function visualTabla(v) {
    return `
      <div class="dem-tarjeta">
        <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">${seguro(v.titulo)}</p>
        <div style="overflow-x:auto"><table class="tabla-dem">
          <thead><tr>${v.columnas.map((c) => `<th>${seguro(c)}</th>`).join("")}</tr></thead>
          <tbody>${v.filas.map((f) => `<tr>${f.map((x) => `<td>${seguro(String(x))}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div>
      </div>`;
  }

  function visualAngulo(v) {
    const largo = 90;
    const cx = 14;
    const cy = 110;
    const radianes = (v.grados * Math.PI) / 180;
    const bx = cx + largo * Math.cos(radianes);
    const by = cy - largo * Math.sin(radianes);
    const r = 34;
    const ax = cx + r;
    const ay = cy;
    const px = cx + r * Math.cos(radianes);
    const py = cy - r * Math.sin(radianes);
    const lectura = v.grados < 20
      ? "Los dos vectores apuntan casi al mismo lado."
      : v.grados < 60
        ? "Los dos vectores apuntan hacia un lado parecido, sin coincidir."
        : "Los dos vectores están cerca de ser perpendiculares: comparten poco.";
    return `
      <div class="dem-tarjeta angulo">
        <svg width="130" height="124" viewBox="0 0 130 124" role="img" aria-label="Ángulo de ${v.grados.toFixed(1)} grados entre los dos vectores">
          <path class="angulo__arco" d="M ${cx} ${cy} L ${ax} ${ay} A ${r} ${r} 0 0 0 ${px} ${py} Z"/>
          <line class="angulo__a" x1="${cx}" y1="${cy}" x2="${cx + largo}" y2="${cy}"/>
          <line class="angulo__b" x1="${cx}" y1="${cy}" x2="${bx}" y2="${by}"/>
        </svg>
        <p>Imagina cada texto como una flecha. <strong>θ = ${v.grados.toFixed(2)}°</strong> es el ángulo entre las dos. ${lectura} A 0° apuntan igual y el coseno vale 1; a 90° no tienen nada en común y vale 0.</p>
      </div>`;
  }

  /* --- búsqueda de patrones: el patrón se desplaza sobre el texto ----------- */

  function visualBusqueda(v) {
    const n = v.texto.length;
    const celdasTexto = [...v.texto].map((c, i) => `<span class="bq__celda" data-t="${i}">${caracter(c)}</span>`).join("");
    const reglas = [...v.texto].map((_, i) => `<span class="bq__regla">${i % 10 === 0 ? i : ""}</span>`).join("");
    const celdasPatron = [...v.patron].map((c, j) => `<span class="bq__celda" data-p="${j}">${caracter(c)}</span>`).join("");
    return `
      <div class="dem-tarjeta" data-busqueda>
        <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">Arriba está el texto y debajo el patrón, es decir, lo que se busca. Verde: la letra coincide. Rojo: no coincide. Gris: no se ha comparado. La casilla azul del texto es donde está mirando el algoritmo, y ␣ es un espacio. ${seguro(v.aviso)}</p>
        <div class="bq" tabindex="0" aria-label="Texto y patrón alineado">
          <div class="bq__pista" style="width:calc(${n} * var(--bq-celda))">
            <div class="bq__fila">${reglas}</div>
            <div class="bq__fila bq__fila--texto">${celdasTexto}</div>
            <div class="bq__fila bq__fila--patron">${celdasPatron}</div>
          </div>
        </div>
        <div class="pd-control">
          <button type="button" class="boton-suave" data-mover="-1" aria-label="Paso anterior">‹</button>
          <input type="range" min="0" max="${v.pasos.length - 1}" value="0" aria-label="Avanzar paso a paso">
          <button type="button" class="boton-suave" data-mover="1" aria-label="Paso siguiente">›</button>
          <span></span>
        </div>
        <div class="pd-explica" aria-live="polite"></div>
      </div>`;
  }

  function activarBusqueda(raiz, v) {
    const pista = raiz.querySelector(".bq");
    const filaPatron = raiz.querySelector(".bq__fila--patron");
    const celdasP = raiz.querySelectorAll("[data-p]");
    const celdasT = raiz.querySelectorAll("[data-t]");
    const rango = raiz.querySelector("input[type=range]");
    const contador = raiz.querySelector(".pd-control span");
    const explica = raiz.querySelector(".pd-explica");
    const ancho = () => celdasT[0].getBoundingClientRect().width;

    function pintar(k) {
      const paso = v.pasos[k];
      filaPatron.style.transform = `translateX(${paso.alineacion * ancho()}px)`;
      celdasP.forEach((celda, j) => {
        const e = paso.estados[j];
        celda.className = `bq__celda${e === "si" ? " si" : e === "no" ? " no" : ""}`;
      });
      celdasT.forEach((celda, i) => {
        celda.className = `bq__celda${i === paso.cursor ? " cursor" : ""}`;
      });
      // El patrón queda a la vista sin animar el desplazamiento: sigue al control 1:1
      const objetivo = paso.alineacion * ancho() + (v.patron.length * ancho()) / 2 - pista.clientWidth / 2;
      pista.scrollLeft = Math.max(0, objetivo);
      explica.innerHTML = `${formula(paso.latex)}<p>${seguro(paso.texto)}</p>`;
      contador.textContent = `paso ${k + 1} de ${v.pasos.length}`;
    }

    rango.addEventListener("input", () => pintar(Number(rango.value)));
    raiz.querySelectorAll("[data-mover]").forEach((boton) => {
      boton.addEventListener("click", () => {
        const k = Math.max(0, Math.min(v.pasos.length - 1, Number(rango.value) + Number(boton.dataset.mover)));
        rango.value = k;
        pintar(k);
      });
    });
    rango.value = v.inicial;
    pintar(v.inicial);
  }

  const VISUALES = {
    busqueda: visualBusqueda,
    matriz: visualMatriz,
    conjuntos: visualConjuntos,
    terminos: visualTerminos,
    palabras: visualPalabras,
    tabla: visualTabla,
    angulo: visualAngulo,
  };

  /* --- lectura del resultado ---------------------------------------------- */

  function lectura(d) {
    let contexto = "";
    if (d.corpus) {
      const { puesto, total, minimo, maximo, valores } = d.corpus;
      const rango = Math.max(maximo - minimo, 1e-9);
      const marcas = valores.map((x) => `<i style="left:${((x - minimo) / rango) * 100}%"></i>`).join("");
      const posicion = Math.max(0, Math.min(100, ((d.similitud - minimo) / rango) * 100));
      contexto = `
        <div class="dem-tarjeta">
          <p style="margin-top:0"><strong>Puesto ${puesto} de ${total}.</strong> Cada línea es uno de los ${total} pares del corpus medido con este algoritmo, y el punto es este par. Un número solo no dice si es alto: hay que verlo frente a los demás.</p>
          <div class="reparto" aria-hidden="true">${marcas}<b style="left:${posicion}%"></b></div>
          <div class="reparto-extremos"><span>mínimo ${minimo.toFixed(4)}</span><span>máximo ${maximo.toFixed(4)}</span></div>
        </div>`;
    }
    return seccion("Cómo se interpreta", `${contexto}<p>${seguro(d.interpretacion)}</p>`,
      "Qué quiere decir este resultado y, sobre todo, qué no quiere decir.");
  }

  function codigo(c) {
    return seccion("Dónde está en el código", `
      <div class="dem-tarjeta">
        <span class="codigo-ruta">${seguro(c.archivo)}</span>
        <ul class="codigo-lista">
          ${c.funciones.map(([f, que]) => `<li><code>${seguro(f)}</code>${seguro(que)}</li>`).join("")}
        </ul>
        <p style="margin:.75rem 0 0;font-size:.9375rem"><strong>Complejidad:</strong> ${seguro(c.tiempo)} en tiempo y ${seguro(c.espacio)} en espacio. ${seguro(c.porque)}.</p>
      </div>`, "Para quien quiera revisar la implementación: el archivo y las funciones que hacen este cálculo. La complejidad dice cuánto crece el trabajo y la memoria cuando los datos crecen.");
  }

  function tusPalabras(d) {
    if (!d.acepta_textos) return "";
    if (d.tipo === "patron") {
      return seccion("Prueba con tus palabras", `
        <div class="dem-tarjeta">
          <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">En el primer campo va el texto y en el segundo lo que quieres buscar. Puedes hacerlo en papel y comparar. Se distinguen mayúsculas de minúsculas.</p>
          <form class="palabras-form" id="form-palabras">
            <input class="campo-texto" name="a" maxlength="300" value="${seguro(d.textos ? d.textos[0] : "ABABDABACDABABCABAB")}" aria-label="Texto" autocomplete="off" spellcheck="false" required>
            <input class="campo-texto" name="b" maxlength="120" value="${seguro(d.textos ? d.textos[1] : "ABABCABAB")}" aria-label="Patrón" autocomplete="off" spellcheck="false" required>
            <button type="submit" class="boton">Calcular</button>
          </form>
        </div>`);
    }
    const dp = d.clave === "levenshtein" || d.clave === "needleman_wunsch";
    const limite = dp ? 40 : 300;
    const inicial = d.textos || (dp ? ["gato", "pato"] : ["the cat sat", "the cat ran"]);
    return seccion("Prueba con tus palabras", `
      <div class="dem-tarjeta">
        <p style="margin-top:0;color:var(--tinta-media);font-size:.9375rem">Escribe dos textos cortos y mira qué da. Puedes hacer el cálculo en papel y comparar.${dp ? " Como son cortos, la tabla sale completa." : ""}</p>
        <form class="palabras-form" id="form-palabras">
          <input class="campo-texto" name="a" maxlength="${limite}" value="${seguro(inicial[0])}" aria-label="Primer texto" autocomplete="off" spellcheck="false" required>
          <input class="campo-texto" name="b" maxlength="${limite}" value="${seguro(inicial[1])}" aria-label="Segundo texto" autocomplete="off" spellcheck="false" required>
          <button type="submit" class="boton">Calcular</button>
        </form>
      </div>`);
  }

  /* ------------------------------------------------------------------- */
  /* La demostración completa                                              */
  /* ------------------------------------------------------------------- */

  function dibujar(d) {
    const avisoPalabras = d.textos ? `
      <div class="aviso-palabras">
        <p>Estás viendo tus palabras, no los abstracts.</p>
        ${retorno ? `<button type="button" class="boton-suave" id="volver-par">${seguro(retorno.etiqueta)}</button>` : ""}
      </div>` : "";

    const abstracts = d.articulos ? d.articulos.map((a) => `
      <details class="abstract">
        <summary>${a.id} · ${seguro(a.titulo)}</summary>
        <p class="abstract__autores">${seguro(a.autores)}</p>
        <p>${seguro(a.texto)}</p>
      </details>`).join("") : "";

    cuerpo.innerHTML = `
      ${avisoPalabras}
      ${cabecera(d)}
      ${seccion("La idea", `<p>${seguro(d.que_es)}</p><p>${seguro(d.idea)}</p>
        <p class="dem-guia">Más abajo está la fórmula, el cálculo paso a paso con estos datos, un dibujo para seguirlo y qué significa el resultado.</p>`)}
      ${seccion("La fórmula", `<div class="dem-tarjeta">${d.formulas.map(formula).join("")}<p class="dem-guia" style="margin-bottom:0">${seguro(d.formula_simple)}</p></div>`,
        "Es la idea anterior escrita en símbolos. Si prefieres, sáltala: el paso a paso hace lo mismo con números reales.")}
      ${seccion("Paso a paso, con estos valores", pasos(d.pasos), "Aquí se hace el cálculo de verdad, en orden. Cada tarjeta es un paso, y los números son los de estos datos.")}
      ${seccion("Para verlo", d.visual.map((v) => VISUALES[v.tipo](v)).join(""), "Lo mismo en tablas y dibujos, para seguirlo con los ojos.")}
      ${lectura(d)}
      ${d.conclusiones ? seccion("Conclusiones", `<div class="dem-tarjeta">${d.conclusiones.map((c) => `<p>${seguro(c)}</p>`).join("")}</div>`, "Lo que se concluye con estos números. Se escribe a partir de los resultados de esta consulta.") : ""}
      ${tusPalabras(d)}
      ${codigo(d.codigo)}
      ${abstracts ? seccion("Los textos comparados", abstracts, "Los dos abstracts completos, por si quieres leerlos.") : ""}`;

    d.visual.forEach((v) => {
      if (v.tipo === "matriz") activarMatriz(cuerpo.querySelector("[data-matriz]"), v);
      if (v.tipo === "busqueda") activarBusqueda(cuerpo.querySelector("[data-busqueda]"), v);
    });

    const form = cuerpo.querySelector("#form-palabras");
    if (form) {
      form.addEventListener("submit", async (evento) => {
        evento.preventDefault();
        const textos = [form.elements.a.value, form.elements.b.value];
        try {
          const resultado = d.tipo === "patron"
            ? await pedir({ algoritmo: d.clave, texto: textos[0], patron: textos[1] }, "/api/demostracion-patron")
            : await pedir({ algoritmo: d.clave, textos });
          dibujar(resultado);
          cuerpo.scrollTop = 0;
        } catch (error) {
          if (error.name !== "AbortError") {
            form.insertAdjacentHTML("afterend", `<p class="error" style="margin-top:.5rem">${seguro(error.message)}</p>`);
          }
        }
      });
    }

    const volver = cuerpo.querySelector("#volver-par");
    if (volver) {
      volver.addEventListener("click", () => retorno.accion());
    }
  }

  async function mostrarAgrupamiento(representacion, enlace, nombre, grupo, disparador) {
    titulo.textContent = nombre;
    familia.textContent = grupo;
    cuerpo.innerHTML = `<p class="hoja__espera" role="status">Agrupando los abstracts…</p>`;
    retorno = null;
    abrir(disparador);
    try {
      dibujar(await pedir({ representacion, enlace }, "/api/demostracion-agrupamiento"));
    } catch (error) {
      if (error.name !== "AbortError") cuerpo.innerHTML = `<p class="error">${seguro(error.message)}</p>`;
    }
  }

  async function mostrarPatron(clave, patron, nombre, disparador) {
    titulo.textContent = nombre;
    familia.textContent = "Búsqueda de patrones";
    cuerpo.innerHTML = `<p class="hoja__espera" role="status">Ejecutando ${seguro(nombre)}…</p>`;
    retorno = { etiqueta: "Volver al corpus", accion: () => mostrarPatron(clave, patron, nombre, null) };
    abrir(disparador);
    try {
      dibujar(await pedir({ algoritmo: clave, patron }, "/api/demostracion-patron"));
    } catch (error) {
      if (error.name !== "AbortError") cuerpo.innerHTML = `<p class="error">${seguro(error.message)}</p>`;
    }
  }

  return { mostrar, mostrarAgrupamiento, mostrarPatron };
})();
