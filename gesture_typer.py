"""gesture_typer.py — Converts a swipe key-path into a word via edit distance."""

import numpy as np
from typing import List, Optional

KEY_POS = {
    "Q":(0,0),"W":(1,0),"E":(2,0),"R":(3,0),"T":(4,0),
    "Y":(5,0),"U":(6,0),"I":(7,0),"O":(8,0),"P":(9,0),
    "A":(0,1),"S":(1,1),"D":(2,1),"F":(3,1),"G":(4,1),
    "H":(5,1),"J":(6,1),"K":(7,1),"L":(8,1),
    "Z":(0,2),"X":(1,2),"C":(2,2),"V":(3,2),"B":(4,2),
    "N":(5,2),"M":(6,2),
}

def _key_dist(a: str, b: str) -> float:
    pa, pb = KEY_POS.get(a.upper()), KEY_POS.get(b.upper())
    if pa is None or pb is None:
        return 3.0
    return float(np.hypot(pa[0]-pb[0], pa[1]-pb[1]))


class GestureTyper:
    def __init__(self, dictionary: Optional[List[str]] = None):
        self._words = dictionary if dictionary else self._default_dict()
        self._words_upper = [w.upper() for w in self._words]

    def match_path(self, path_keys: List[str]) -> Optional[str]:
        if len(path_keys) < 2:
            return None
        path_upper = [k.upper() for k in path_keys]
        # deduplicate consecutive
        deduped = [path_upper[0]]
        for k in path_upper[1:]:
            if k != deduped[-1]:
                deduped.append(k)
        first, last = deduped[0], deduped[-1]
        scores = []
        for word_u in self._words_upper:
            if len(word_u) < 2:
                continue
            if word_u[0] != first or word_u[-1] != last:
                continue
            score = self._edit_dist(deduped, list(word_u))
            scores.append((score / max(len(word_u), len(deduped)), word_u))
        if not scores:
            return None
        scores.sort(key=lambda x: x[0])
        return scores[0][1].lower()

    def _edit_dist(self, path: List[str], word: List[str]) -> float:
        n, m = len(path), len(word)
        dp = np.full((n+1, m+1), np.inf, dtype=np.float32)
        dp[0, 0] = 0.0
        for i in range(1, n+1): dp[i, 0] = dp[i-1, 0] + 0.5
        for j in range(1, m+1): dp[0, j] = dp[0, j-1] + 1.0
        for i in range(1, n+1):
            for j in range(1, m+1):
                dp[i, j] = min(
                    dp[i-1, j-1] + _key_dist(path[i-1], word[j-1]),
                    dp[i-1, j]   + 0.5,
                    dp[i,   j-1] + 1.0,
                )
        return float(dp[n, m])

    @staticmethod
    def _default_dict() -> List[str]:
        return [
            "able","about","above","accept","account","achieve","across","action",
            "actually","add","address","after","again","age","ago","agree","ahead",
            "air","all","allow","already","also","always","am","among","and",
            "another","answer","any","anyone","app","apple","apply","are","area",
            "around","art","ask","at","away","back","ball","be","because","been",
            "before","best","better","between","big","black","blue","both","break",
            "bring","build","but","buy","by","call","can","car","care","carry",
            "case","cause","change","check","child","city","class","close","cold",
            "come","control","could","country","create","cut","data","day","dead",
            "deal","deep","delete","design","did","different","do","does","done",
            "down","drive","drop","each","early","earth","easy","eat","else","end",
            "enter","even","ever","every","example","expect","eye","face","fact",
            "fall","far","fast","feel","few","field","file","find","first","follow",
            "for","force","form","forward","free","from","front","full","fun",
            "future","game","get","give","go","good","great","green","grow","guide",
            "hand","hard","have","head","help","here","high","him","his","hold",
            "home","hope","how","human","idea","if","image","important","in",
            "include","info","into","is","it","its","job","join","just","keep",
            "key","know","knowledge","large","last","later","learn","left","let",
            "level","life","light","like","line","list","live","local","long",
            "look","low","made","main","make","many","may","me","mean","meet",
            "model","more","most","move","much","my","name","need","network","new",
            "next","night","no","note","now","of","off","often","ok","on","one",
            "only","open","or","other","out","over","own","part","pass","path",
            "people","place","plan","play","point","power","press","problem","put",
            "quick","quite","read","real","red","report","result","return","right",
            "run","same","save","say","screen","search","see","seem","send","set",
            "share","should","show","side","since","size","small","some","sort",
            "space","start","step","still","stop","store","strong","such","sure",
            "system","take","tell","text","than","that","the","their","them","then",
            "there","they","think","this","through","time","to","today","too","top",
            "try","turn","type","under","up","use","user","very","view","voice",
            "want","was","way","we","well","what","when","where","which","white",
            "who","why","will","with","word","work","world","write","year","yes",
            "you","your","zero","zoom","hello","world","help","python","code",
            "keyboard","finger","gesture","swipe","click","screen","camera","track",
        ]
