-- Migrazione 002: aggiunge colonna image_variants + variants_ready allo status check
-- Eseguire su Supabase SQL Editor

-- 1. Colonna per le 3 varianti generate
ALTER TABLE social_content
ADD COLUMN IF NOT EXISTS image_variants JSONB DEFAULT '[]'::jsonb;

-- Struttura di ogni elemento nell'array:
-- {
--   "index": 0,           -- 0=minimal, 1=editorial, 2=intimate
--   "direction": "minimal",
--   "feed_url": "https://...",
--   "story_url": "https://..."
-- }

-- 2. Aggiunge variants_ready al check constraint sullo status
ALTER TABLE social_content
DROP CONSTRAINT social_content_status_check;

ALTER TABLE social_content
ADD CONSTRAINT social_content_status_check
CHECK (status IN (
  'waiting_material',
  'material_ready',
  'brief_ready',
  'generating',
  'variants_ready',
  'draft',
  'approved',
  'rejected',
  'published'
));
