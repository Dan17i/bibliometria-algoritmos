"""
Lista de stopwords en inglés, escrita a mano (sin NLTK ni spaCy).
------------------------------------------------------------------
Son palabras funcionales (artículos, preposiciones, pronombres, auxiliares,
conjunciones) que aparecen en casi cualquier texto y casi no aportan significado
temático. Se usan para quitarlas antes de promediar vectores de palabras: sin
ellas el promedio se parece mucho entre cualquier par de abstracts.

No se incluyen palabras de contenido ("ai", "learning", "review", "studies"...),
ni negaciones como "not" o "no", que cambian el sentido de la frase; tampoco
"first", "second", etc. Las letras sueltas (s, t, d, ll, re, ve, m) son los restos
de contracciones y posesivos ("AI's" -> "ai" + "s").
"""

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at
    be because been before being below between both but by
    can can't cannot could couldn't
    did didn't do does doesn't doing don't down during
    each
    few for from further
    had hadn't has hasn't have haven't having he he'd he'll he's her here here's
    hers herself him himself his how how's
    i i'd i'll i'm i've if in into is isn't it it's its itself
    let's
    me more most mustn't my myself
    nor of off on once only or other ought our ours ourselves out over own
    same shan't she she'd she'll she's should shouldn't so some such
    than that that's the their theirs them themselves then there there's these
    they they'd they'll they're they've this those through to too
    under until up
    very
    was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's will with won't would
    wouldn't
    you you'd you'll you're you've your yours yourself yourselves
    also however thus therefore whereof whereas within without
    may might must shall
    s t d ll re ve m
    """.split()
)


def remove_stopwords(tokens: list[str], stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """
    Devuelve los tokens que NO son stopwords, conservando el orden y las repeticiones.

    Complejidad temporal: O(T), un acceso O(1) al conjunto por token.
    Complejidad espacial: O(T) por la lista resultante.
    """
    return [t for t in tokens if t not in stopwords]
