# Rekonštrukcia vysokých hodnôt merania a opis prevádzkových profilov

## 1. Pochopenie problému a cieľa

**Určenie priority opakovaných meraní.** Ak jedna nameraná hodnota nie je k dispozícii, ktoré pozorovania má obsluha premerať ako prvé? Táto práca predbežne definuje vysokú hodnotu podmienkou `f_09 > 16` a ako vstupy používa zvyšných 28 meraní. Pri vyhodnotení poskytuje zaznamenaná hodnota `f_09` správnu odpoveď; na jej predikciu sa nesmie použiť ani samotný atribút, ani atribúty z neho odvodené, ani identifikátor riadka. Hranica je 90. percentil trénovacej množiny, nie technický bezpečnostný limit.

Predpokladáme, že možno opätovne premerať 10 zo 100 pozorovaní. Kritériom je podiel všetkých vysokých hodnôt, ktoré sa podarí zachytiť výberom týchto 10 pozorovaní (senzitivita pri 10 % kapacite na premeranie). Jednoduchý referenčný postup uprednostní nízke hodnoty `f_08`. Postupy možno pri rovnakej kapacite porovnať pomocou vzťahu `počet premeraní × cena premerania + počet nezachytených vysokých hodnôt × náklady na jeden nezachytený prípad`. Ani jeden jednotkový náklad nepoznáme, preto neuvádzame finančné úspory.

**Identifikácia prevádzkových profilov na účely monitorovania.** Druhým cieľom je nájsť stabilné skupiny, opísať merania, ktorými sa odlišujú, a posúdiť, či sa medzi nimi líši podiel vysokých hodnôt. Preto `f_09` pri tvorbe zhlukov nepoužijeme. Neskorším experimentom možno overiť, do akej miery jeden klasifikátor na nových pozorovaniach reprodukuje konsenzus viacerých zhlukovacích metód a aké sú jeho výpočtové náklady. Skutočné označenia riadiacich stavov nepoznáme.

## 2. Pochopenie dát

[Súťažné dáta](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/data) obsahujú 98 000 simulovaných výrobných pozorovaní s jedinečnými identifikátormi riadkov. Merania sú v súbore `data.csv`; `sample_submission.csv` je šablóna výstupu, nie ďalšia tabuľka pozorovaní na spojenie. Z 29 anonymných meraní sedem (`f_07`–`f_13`) nadobúda celočíselné hodnoty a 22 je spojitých. Ich jednotky ani fyzikálny význam nie sú uvedené.

Analýza zachováva pôvodné prieskumné rozdelenie: pozície riadkov v pôvodnom poradí sa premiešajú generátorom NumPy `default_rng(42)`. Prvých 62 720 riadkov tvorí trénovaciu množinu, ďalších 15 680 validačnú a zvyšných 19 600 záverečnú testovaciu množinu. Všetky štatistiky a grafy v tejto časti vychádzajú len z trénovacích riadkov. V trénovacej množine nechýbajú namerané hodnoty a žiadne dva riadky nemajú zhodné hodnoty všetkých meraní. [Tabuľka 1: Súhrn atribútov](tables/attribute_summary.csv) uvádza kvartily aj počty hodnôt mimo hraníc určených 1,5-násobkom medzikvartilového rozpätia. Pri `f_09` leží mimo týchto hraníc 2 148 hodnôt. Ponechávame ich: práve extrémne hodnoty má prediktívna úloha pomôcť zachytiť.

![Pearsonove korelácie medzi 29 meraniami v trénovacej množine](figures/attribute_correlations.png)

**Obrázok 1.** Najsilnejší párový lineárny vzťah je medzi `f_08` a `f_09` (Pearsonov koeficient −0,361; Spearmanov koeficient −0,343). Záporná korelácia je aj medzi `f_09` a `f_12` (Pearsonov koeficient −0,297). [Tabuľka 2: Vzťahy medzi atribútmi](tables/attribute_relationships.csv) uvádza oba koeficienty pre všetkých 406 rôznych dvojíc.

![Podiel vysokých hodnôt v štyroch intervaloch f_08](figures/f09_high_rate_by_f08.png)

**Obrázok 2.** Podmienku `f_09 > 16` spĺňa 6 106 trénovacích riadkov (9,74 %). Pri `f_08 ≤ 4` tento podiel stúpa na 21,97 % (20 828 riadkov); pri `f_08 ≥ 10` klesá na 2,03 % (15 453 riadkov). Hranice intervalov vychádzajú z trénovacích kvartilov `f_08`; v dôsledku zhôd na celočíselných hraniciach majú skupiny rôznu veľkosť. Približne jedenásťnásobný rozdiel odôvodňuje otestovanie `f_08` ako referenčného pravidla na určenie priority premerania. Nedokazuje však príčinnú súvislosť ani neidentifikuje poruchu stroja.

## 3. Príprava dát a referenčné pravidlá

Cieľová premenná má hodnotu 1, ak `f_09 > 16`, a 0 vo všetkých ostatných prípadoch vrátane hodnoty 16. Zvyšných 28 meraní používame ako vstupy a `id` ako identifikátor riadka. V trénovacej množine nebolo potrebné dopĺňať chýbajúce hodnoty. [Tabuľka 3: Významnosť vstupných atribútov](tables/feature_relevance.csv) uvádza Spearmanov korelačný koeficient každého vstupu voči binárnej cieľovej premennej v trénovacej množine. Najsilnejšie jednotlivé vzťahy sú záporné: `f_08` (−0,287), `f_13` (−0,261) a `f_11` (−0,229). Slabá individuálna korelácia nevylučuje užitočný spoločný účinok, preto na tomto základe vstupy nevyraďujeme.

Dve pravidlá založené na jednom meraní (uprednostnenie najnižších hodnôt `f_08` alebo `f_12`) porovnávame s jedným odvodeným skóre. Percentil každého merania vyjadruje podiel trénovacích hodnôt, ktoré sú menšie alebo rovné pozorovanej hodnote. Kombinované skóre je aritmetický priemer oboch percentilov. Nízke skóre uprednostní pozorovania, pri ktorých majú obe merania sklon nadobúdať nízke hodnoty. Pri výpočte skóre pre validačné riadky používame nezmenené trénovacie rozdelenie.

### Validácia pri pevnej kapacite na premeranie

Každé pravidlo vyberie presne 1 568 z 15 680 validačných pozorovaní. Pri rovnosti skóre rozhoduje spoločné náhodné poradie pôvodne zoradených identifikátorov (seed 42), nezávislé od cieľovej hodnoty. Z 1 526 vysokých hodnôt zachytilo samotné `f_08` celkovo 469 (senzitivita 30,73 %), samotné `f_12` zachytilo 225 (14,74 %) a kombinované skóre zachytilo **567 (37,16 %)**. Spojením meraní sa pri rovnakom počte premeraní zachytilo **o 98 vysokých hodnôt viac** než pomocou samotného `f_08`.

[Tabuľka 4: Porovnanie referenčných pravidiel](tables/baseline_comparison.csv) uvádza aj podiel vysokých hodnôt medzi vybranými riadkami (precision) a jeho pomer k podielu v celej validačnej množine (lift). Pri kombinovanom pravidle bolo vysokých 36,16 % vybraných hodnôt oproti 9,73 % v celej validačnej množine. Pravidlo teda zachytilo **3,72-násobok očakávaného počtu pri náhodnom výbere**. Kombinované pravidlo bude slúžiť na porovnanie s klasifikátormi. Testovacia množina zostáva vyhradená na ich záverečné vyhodnotenie.

## 4. Prediktívne modelovanie

Tri klasifikátory používajú 28 pôvodných vstupných meraní. Logistická regresia modeluje aditívny vplyv vstupov na logaritmus šance. Jej vstupy sme štandardizovali pomocou priemeru a smerodajnej odchýlky odhadnutých vždy len na trénovacej časti daného rozdelenia. Rozhodovací strom dokáže vyjadriť prahové pravidlá a interakcie bez zmeny mierky. Náhodný les priemeruje pravdepodobnosti z 96 stromov učených na vzorkách získaných výberom s opakovaním. Pri každom vetvení posudzuje náhodnú podmnožinu vstupov. Kombinované percentilové skóre zostáva samostatným referenčným pravidlom.

V trénovacej množine sme použili trojnásobnú stratifikovanú krížovú validáciu s náhodným premiešaním (seed 42). Pri logistickej regresii sme skúšali hodnoty parametra inverznej sily regularizácie `C` 0,01; 0,1 a 1. Pri strome sme kombinovali maximálne hĺbky 4 a 8 s minimálnym počtom 20 alebo 100 pozorovaní v liste. Pri lese sme kombinovali minimálne počty 5 a 25 pozorovaní v liste s dvoma počtami vstupov posudzovaných pri vetvení: odmocninou počtu vstupov alebo polovicou vstupov. Pre každý klasifikátor sme vybrali nastavenie s najvyššou priemernou senzitivitou pri premeraní prvých 10 % pozorovaní z každej vynechanej časti. Potom sme ho natrénovali na celej trénovacej množine. [Tabuľka 5: Optimalizácia parametrov](tables/classifier_tuning.csv) uvádza priemer a smerodajnú odchýlku zo všetkých troch častí pre každé nastavenie.

Pri výbere modelu sme validačné riadky zoradili podľa odhadnutej pravdepodobnosti `f_09 > 16`. Použili sme rovnakú kapacitu na premeranie aj rovnaký postup pri zhode skóre ako pri referenčnom pravidle. Víťaza sme vybrali pred vyhodnotením testovacej množiny. [Tabuľka 6: Porovnanie prediktívnych metód](tables/classifier_comparison.csv) uvádza výsledky všetkých metód na všetkých troch množinách. Celková presnosť, senzitivita, špecificita, podiel skutočne vysokých hodnôt medzi vybranými (precision), lift a počty v kontingenčnej tabuľke vychádzajú z výberu na premeranie; ROC-AUC využíva celé poradie. Trénovacie metriky opisujú výsledky na dátach použitých pri učení.

## 5. Vyhodnotenie prediktívnej úlohy

Pri 1 568 premeraniach vo validačnej množine náhodný les zachytil 871 vysokých hodnôt, logistická regresia 843, strom 806 a referenčné pravidlo 567. Vybraný les pri každom vetvení posudzoval 14 z 28 vstupov a vyžadoval aspoň 25 trénovacích pozorovaní v liste.

V testovacej množine bolo z 19 600 hodnôt vysokých 1 854. Pri 1 960 premeraniach náhodný les zachytil **1 058** vysokých hodnôt, kým referenčné pravidlo zachytilo **703**: **o 355 viac pri rovnakých nákladoch na premeranie**. Les nezachytil 796 vysokých hodnôt a na premeranie vybral 902 hodnôt, ktoré neboli vysoké. Senzitivita bola 57,07 %, špecificita 94,92 %, celková presnosť 91,34 % a ROC-AUC 0,901. Nižšia senzitivita na testovacej množine než na trénovacej (67,64 %) ukazuje, že samotný trénovací výsledok by nadhodnotil očakávaný počet zachytených prípadov.
