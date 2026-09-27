# Actions only the user can take

Steps the build cannot do for you, with where to go and what to click. When one is done, tell the
session the words to record (they go into the licence register or the plan with `--user-confirmed`).

## 1. entscheidsuche.ch: confirm the terms for the Swiss Federal Supreme Court decisions (before P4)

**Why.** The `bger` collector (P3-L) takes the Federal Supreme Court's decisions from entscheidsuche.ch,
run by the non-profit Verein entscheidsuche.ch (Bern). The decisions themselves are free of copyright
(Art. 5 Abs. 1 lit. c URG). The association's own terms decide how its service may be used. P4 will
download its 109,262 German decisions (counted 2026-09-27; about 1.5 GB of text, at one request per
second roughly 30 hours).

**What the published terms say** (read on 2026-09-27; you confirm by reading them yourself):

- The "Daten" page says the collected data "stehen alle Interessierten zur freien Benutzung und
  Weiterbearbeitung zur Verfügung". It asks to be named as the source. It also asks services that
  process the data further for a voluntary contribution.
- The API documentation (3 July 2022) says: "You are free to use the API and the cases in any way you
  want. However, we ask you to be kind to our server and not to generate too much load … mention
  entscheidsuche.ch as your source and support entscheidsuche.ch with a donation if you use the API
  commercially – however this is not a legal requirement."

**Steps.**

1. Open **https://entscheidsuche.ch/dataUsage** in a browser (the page headed "Daten"; from the start
   page it is the data-usage entry of the site's menu).
2. Read the section **"Weiterverwendung der Daten"**. Check that it still says free use and further
   processing, with the source named.
3. Open **https://entscheidsuche.ch/pdf/EntscheidsucheAPI.pdf**. Page 1 has the terms quoted above.
   Page 10 ("Blockliste") explains that withdrawn decisions are listed in
   `https://entscheidsuche.ch/docs/Blockliste.json`; the collector already skips them.
4. Decide two things and tell the session:
   - **Attribution line.** For the model card and README, for example: "Decisions of the Swiss Federal
     Supreme Court obtained via entscheidsuche.ch."
   - **Donation.** If the model will be used commercially, the association asks for a voluntary donation.
     It is not a legal requirement. Say yes or no; the About page, https://entscheidsuche.ch/about, has
     the TWINT and bank details under "Unterstützen Sie uns".
5. **Recommended, as a courtesy before the large download.** E-mail the board at
   **info@entscheidsuche.ch** (Jörn Erbguth, Daniel Kettiger, Claudia Schreiber). A draft, which you can
   adapt:

   > Betreff: Nutzung von entscheidsuche.ch für ein Forschungs-Sprachmodell (Bundesgericht, Deutsch)
   >
   > Guten Tag
   >
   > Wir bauen ein deutschsprachiges juristisches Sprachmodell, das Antworten mit Fundstellen belegt,
   > und möchten dafür die deutschsprachigen Entscheide des Bundesgerichts (Sammlung CH_BGer) über Ihre
   > Suchschnittstelle und die Dateien unter /docs/CH_BGer/ beziehen: etwa 109'000 Entscheide, eine
   > Anfrage pro Sekunde, mit dem User-Agent "lexhybrid-corpus/0.1". Die Blockliste wird beachtet,
   > entscheidsuche.ch wird als Quelle genannt. Gibt es ein Zeitfenster oder einen Weg, der Ihren
   > Server weniger belastet, den wir bevorzugen sollten?
   >
   > Freundliche Grüsse
   > [Name, Organisation, E-Mail]

6. Tell the session in one line, for example: *"entscheidsuche terms confirmed on <date>; attribution
   '<line>'; donation yes/no; e-mail sent on <date> (reply: …)"*. The session writes that into the
   `bger` row of `Docs/CORPUS_LICENCE_REGISTER.md` and removes the *to confirm* mark.

## 2. Flair NER model weights: a licence statement before any release (not blocking)

The model card of `flair/ner-german-legal` states no licence (checked in P3-R). It ships with Flair
(MIT) as Flair's default German legal model. Using it to scrub names from the corpus is preprocessing;
only its placeholders reach the corpus. Before a commercial release, ask the Flair maintainers:
https://github.com/flairNLP/flair/issues → "New issue" → ask for the licence of the `ner-german-legal`
weights. Record the answer in the models table of the register.
