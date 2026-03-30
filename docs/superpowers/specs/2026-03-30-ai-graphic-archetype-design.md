# AI Graphic Archetype — Design Spec
_Data: 2026-03-30_

## Obiettivo

Aggiungere al piano editoriale settimanale un post completamente generato dall'AI (`ai_graphic`), senza necessità di caricare foto. L'AI decide concept, testo e grafica. L'estetista riceve il post già pronto nel frontend e deve solo approvarlo o rifiutarlo.

---

## Scopo e vincoli

- **1 post ai_graphic per settimana**, aggiunto automaticamente insieme ai post basati su appuntamenti
- **Nessun upload di foto** richiesto all'estetista
- **Revisione obbligatoria**: il post arriva in stato `draft`, l'estetista approva prima che vada in calendario
- **Rotazione fissa** tra 4 categorie di contenuto

---

## Architettura

### File modificati (solo 2)

**`src/social/gemini_social.py`**
- Nuova funzione: `generate_ai_graphic_post(tenant: dict) -> dict`
  - Legge profilo brand tenant
  - Calcola categoria corrente dalla rotazione
  - Chiama Gemini testo → `caption_text`, `hashtags`, `visual_concept`
  - Chiama Gemini immagine → genera grafica senza foto in input
  - Ritorna dict completo con tutti i campi del record

**`src/social/content_pipeline.py`**
- In fondo a `run_weekly_pipeline`, dopo i post appuntamenti:
  - Chiama `generate_ai_graphic_post(tenant)`
  - Carica le immagini su Supabase Storage
  - Salva record con `status="draft"`, immagini già popolate

### File NON modificati
- `supabase_queries.py` — `create_social_content` già funziona
- `router.py` — endpoint approve/reject già gestisce status `draft → approved`
- Schema DB — nessuna migrazione necessaria

---

## Rotazione categorie

4 categorie in ciclo fisso:

| # | Categoria | Descrizione |
|---|-----------|-------------|
| 1 | `tip_beauty` | Consiglio pratico su cura/bellezza |
| 2 | `spotlight` | Mette in vetrina un servizio del centro |
| 3 | `stagionale` | Legato al periodo dell'anno / tendenza |
| 4 | `ispirazione` | Citazione motivazionale stile brand |

**Tracking:** `tenant.social_profile.last_ai_graphic_category` (stringa con nome categoria).
Ogni domenica: legge l'ultimo valore → avanza al successivo → aggiorna il campo.
Se il campo è vuoto (primo avvio), parte da `tip_beauty`.

Per `spotlight`: l'AI legge i servizi del centro da Supabase e sceglie quello non recentemente evidenziato (stesso meccanismo rotazione archetypes esistente — query last 8 records per service).

---

## Flusso status

```
Pipeline domenica
      ↓
generate_ai_graphic_post() → immagine generata immediatamente
      ↓
status: "draft"  ←  immagine già visibile nel frontend
      ↓
Estetista approva
      ↓
status: "approved" → in calendario
```

Campi del record salvato:
- `archetype = "ai_graphic"`
- `status = "draft"`
- `appointment_id = null`
- `service_id = null` (tranne spotlight, dove viene popolato)
- `material_checklist = []`
- `photos_input = []`
- `image_url_feed` e `image_url_story` già popolati
- `caption_text` e `hashtags` già generati

---

## Generazione immagine

Il prompt per Gemini Image è costruito in 4 parti:

1. **Brand identity**: colori (primary, secondary, accent, background), stile visivo, font style estratti da `social_profile`
2. **Formato**: grafica di design con testo (NO foto realistiche di persone — evita uncanny valley)
3. **Contenuto per categoria**:
   - `tip_beauty`: layout testo principale + sottotitolo + icona decorativa
   - `spotlight`: nome trattamento in evidenza + 2-3 benefici + CTA
   - `stagionale`: elemento stagionale (colori/elementi del periodo) + messaggio brand
   - `ispirazione`: citazione centrata + elemento grafico minimal + firma brand
4. **Specifiche tecniche**: feed 1:1 e story 9:16 con layout adattato

---

## Test plan

Prima del deploy in produzione:
1. Eseguire script di test locale che legge il profilo Amati da Supabase
2. Generare 2-3 post di categorie diverse (es. tip_beauty, spotlight, ispirazione)
3. Salvare le immagini generate localmente e mostrarle all'utente
4. Solo se qualità approvata → commit + push su master

---

## Cosa NON è incluso in questo scope

- Possibilità di rigenerare il post ai_graphic dal frontend (può essere aggiunto in futuro)
- Più di 1 post ai_graphic a settimana
- Scelta manuale della categoria da parte dell'estetista
