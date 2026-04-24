"""prediction_engine.py — Prefix Trie + frequency + bigram word prediction."""

from typing import List, Optional, Dict

# ── Frequency word list ────────────────────────────────────────────────────────
WORD_FREQ: Dict[str, int] = {}
_RAW = """
the:1000000 be:900000 to:800000 of:750000 and:700000 a:650000 in:600000
that:550000 have:500000 it:480000 for:460000 not:440000 on:420000 with:400000
he:380000 as:360000 you:340000 do:320000 at:300000 this:290000 but:280000
his:270000 by:260000 from:250000 they:240000 we:230000 say:220000 her:210000
she:200000 or:190000 an:180000 will:175000 my:170000 one:165000 all:160000
would:155000 there:150000 their:145000 what:140000 so:135000 up:130000
out:125000 if:120000 about:115000 who:110000 get:105000 which:100000 go:99000
me:98000 when:97000 make:96000 can:95000 like:94000 time:93000 no:92000
just:91000 him:90000 know:89000 take:88000 people:87000 into:86000 year:85000
your:84000 good:83000 some:82000 could:81000 them:80000 see:79000 other:78000
than:77000 then:76000 now:75000 look:74000 only:73000 come:72000 its:71000
over:70000 think:69000 also:68000 back:67000 after:66000 use:65000 new:56000
want:55000 because:54000 any:53000 give:51000 day:50000 most:49000 us:48000
great:47000 between:46000 need:45000 hand:42000 high:41000 place:40000
free:38000 real:37000 life:36000 open:33000 next:30000 white:29000 always:20000
care:9000 above:5000 ever:4800 feel:4000 talk:3800 soon:3400 body:3200
family:2900 door:2300 black:2100 short:2000 class:1800 question:1600
problem:960 pass:920 top:900 space:870 best:850 better:830 true:820
step:770 early:760 fast:700 dark:500 note:480 plan:460 star:440
done:360 front:310 week:290 green:260 quick:240 warm:210 mind:160
clear:140 fact:110 full:93 blue:91 deep:87 system:83 test:81
record:80 check:67 game:66 hot:63 yes:56 fill:54 ball:48
wave:46 drop:45 heart:44 dance:40 arm:37 speak:30 matter:26
hello:5000 world:4900 help:4800 python:4700 code:4600 data:4500
type:4400 text:4300 keyboard:4200 finger:4100 hand:4000 gesture:3900
swipe:3800 click:3700 press:3600 hover:3500 screen:3400 camera:3300
track:3200 detect:3100 model:3000 learn:2900 predict:2700 input:2600
output:2500 function:2400 import:2200 return:2100 print:2000 range:1900
index:1800 value:1700 string:1600 number:1500 array:1400 list:1300
send:120 show:110 toggle:100 enable:95 reset:85 delete:75 update:65
fetch:60 post:55 connect:35 read:20 write:15 run:61000 work:61000
first:60000 well:59000 way:58000 even:57000 large:44000 often:43000
"""
for pair in _RAW.split():
    if ":" in pair:
        w, f = pair.rsplit(":", 1)
        WORD_FREQ[w.strip().lower()] = int(f)


# ── Trie ───────────────────────────────────────────────────────────────────────
class TrieNode:
    __slots__ = ("children", "word", "freq")
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.word:     Optional[str] = None
        self.freq:     int = 0


class PredictionEngine:
    def __init__(self):
        self._root = TrieNode()
        for word, freq in WORD_FREQ.items():
            self._insert(word, freq)
        self._bigrams: Dict[str, Dict[str, int]] = {}
        self._build_bigrams()

    def predict(self, prefix: str, context: Optional[str] = None,
                n: int = 5) -> List[str]:
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        results = self._search_prefix(prefix)
        if context:
            ctx = context.lower()
            bg  = self._bigrams.get(ctx, {})
            results.sort(key=lambda w: bg.get(w, 0)*10 + WORD_FREQ.get(w, 0),
                         reverse=True)
        else:
            results.sort(key=lambda w: WORD_FREQ.get(w, 0), reverse=True)
        return results[:n]

    def add_word(self, word: str, freq: int = 100):
        w = word.lower().strip()
        if w:
            WORD_FREQ[w] = max(WORD_FREQ.get(w, 0), freq)
            self._insert(w, WORD_FREQ[w])

    def _insert(self, word: str, freq: int):
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.word = word
        node.freq = freq

    def _search_prefix(self, prefix: str) -> List[str]:
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: List[str] = []
        self._dfs(node, results)
        return results

    def _dfs(self, node: TrieNode, out: List[str], limit: int = 50):
        if len(out) >= limit:
            return
        if node.word is not None:
            out.append(node.word)
        for child in node.children.values():
            if len(out) >= limit:
                break
            self._dfs(child, out, limit)

    def _build_bigrams(self):
        phrases = [
            "i am","i have","i will","i can","you are","you have","you can",
            "it is","it was","we are","we have","we will","they are","they have",
            "there is","there are","this is","this was","that is","that was",
            "the best","the first","the last","the next","the same","the new",
            "a good","a great","a new","a big","in the","in a","on the","on a",
            "at the","to the","to be","to do","for the","for a","with the",
            "can you","can i","do you","will you","have you","hello world",
            "hello there","thank you","good morning","good night","good luck",
            "how are","how do","what is","what are","where is","when is","why is",
            "python code","data type","function call","hand gesture","finger tip",
        ]
        for phrase in phrases:
            words = phrase.split()
            for i in range(len(words)-1):
                w1, w2 = words[i], words[i+1]
                if w1 not in self._bigrams:
                    self._bigrams[w1] = {}
                self._bigrams[w1][w2] = self._bigrams[w1].get(w2, 0) + 1
