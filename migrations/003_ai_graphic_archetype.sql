-- migrations/003_ai_graphic_archetype.sql
-- Aggiunge 'ai_graphic' al CHECK constraint archetype in social_content

ALTER TABLE public.social_content
DROP CONSTRAINT IF EXISTS social_content_archetype_check;

ALTER TABLE public.social_content
ADD CONSTRAINT social_content_archetype_check
CHECK (archetype IN (
    'before_after', 'editorial', 'promo',
    'educational', 'behind_scenes', 'retention', 'ai_graphic'
));
