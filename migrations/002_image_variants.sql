-- Migrazione 002: aggiunge colonna image_variants per il sistema a 3 varianti
-- Eseguire su Supabase SQL Editor

ALTER TABLE social_content
ADD COLUMN IF NOT EXISTS image_variants JSONB DEFAULT '[]'::jsonb;

-- Struttura di ogni elemento nell'array:
-- {
--   "index": 0,           -- 0=minimal, 1=editorial, 2=intimate
--   "direction": "minimal",
--   "feed_url": "https://...",
--   "story_url": "https://..."
-- }
