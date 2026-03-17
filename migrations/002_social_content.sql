-- ============================================================
-- Migration 002: Social Content AI
-- Esegui in Supabase SQL Editor
-- ============================================================

-- 1. Nuova tabella social_content
-- ============================================================
CREATE TABLE IF NOT EXISTS public.social_content (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    appointment_id          UUID REFERENCES public.appointments(id) ON DELETE SET NULL,
    service_id              UUID REFERENCES public.services(id) ON DELETE SET NULL,

    -- Pianificazione
    week_start              DATE NOT NULL,
    week_end                DATE NOT NULL,
    scheduled_date          DATE,

    -- Tipo contenuto
    platform                TEXT NOT NULL DEFAULT 'both'
                            CHECK (platform IN ('instagram', 'facebook', 'both')),
    content_type            TEXT NOT NULL DEFAULT 'post'
                            CHECK (content_type IN ('post', 'story', 'reel', 'carousel')),
    archetype               TEXT NOT NULL DEFAULT 'editorial'
                            CHECK (archetype IN (
                                'before_after', 'editorial', 'promo',
                                'educational', 'behind_scenes', 'retention'
                            )),

    -- Materiale richiesto (checklist generata da Gemini)
    material_checklist      JSONB DEFAULT '[]',
    -- Struttura: [{ id, label, instructions, required, uploaded_url }]

    -- Input estetista
    photos_input            TEXT[] DEFAULT '{}',
    client_consent          TEXT DEFAULT 'details_only'
                            CHECK (client_consent IN ('with_face', 'details_only', 'no_client')),
    estetista_notes         TEXT,

    -- Output AI — testo
    visual_brief            TEXT,       -- descrizione piano grafico PRIMA della generazione
    visual_brief_override   TEXT,       -- eventuale override dell'estetista
    caption_text            TEXT,
    hashtags                TEXT[] DEFAULT '{}',
    reel_script             TEXT,
    selection_rationale     TEXT,       -- perché Gemini ha scelto questo appuntamento

    -- Output AI — immagini (URL Supabase Storage)
    image_url_feed          TEXT,       -- 1:1 per feed
    image_url_story         TEXT,       -- 9:16 per story

    -- Workflow status
    status                  TEXT NOT NULL DEFAULT 'planned'
                            CHECK (status IN (
                                'planned',          -- sistema ha pianificato
                                'waiting_material', -- in attesa foto estetista
                                'material_ready',   -- foto caricate, genera brief
                                'brief_ready',      -- brief pronto, in attesa approvazione
                                'generating',       -- Gemini sta generando immagine
                                'draft',            -- tutto pronto, in approvazione
                                'approved',         -- approvato, in calendario
                                'published',        -- pubblicato (futuro)
                                'rejected'          -- rifiutato
                            )),

    -- Timestamp workflow
    material_uploaded_at    TIMESTAMPTZ,
    brief_generated_at      TIMESTAMPTZ,
    brief_approved_at       TIMESTAMPTZ,
    image_generated_at      TIMESTAMPTZ,
    approved_at             TIMESTAMPTZ,
    published_at            TIMESTAMPTZ,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indici
CREATE INDEX IF NOT EXISTS idx_social_content_tenant_week
    ON public.social_content(tenant_id, week_start);

CREATE INDEX IF NOT EXISTS idx_social_content_tenant_status
    ON public.social_content(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_social_content_scheduled
    ON public.social_content(tenant_id, scheduled_date);

-- Unique: un contenuto per appuntamento per settimana per archetype
CREATE UNIQUE INDEX IF NOT EXISTS idx_social_content_unique_appt_week
    ON public.social_content(appointment_id, week_start, archetype)
    WHERE appointment_id IS NOT NULL;

-- Trigger updated_at
CREATE OR REPLACE FUNCTION public.update_social_content_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_social_content_updated_at ON public.social_content;
CREATE TRIGGER trg_social_content_updated_at
    BEFORE UPDATE ON public.social_content
    FOR EACH ROW EXECUTE FUNCTION public.update_social_content_updated_at();


-- 2. Aggiunte a tabella tenants
-- ============================================================
ALTER TABLE public.tenants
    ADD COLUMN IF NOT EXISTS social_profile     JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS content_api_key    TEXT;

-- Struttura social_profile:
-- {
--   "tone_of_voice": "caldo e professionale",
--   "style": "minimal",
--   "brand_keywords": ["naturale", "cura"],
--   "content_frequency": 3,
--   "platforms": ["instagram", "facebook"],
--   "generation_day": "sunday",
--   "generation_hour": 18,
--   "style_reference": "Charlotte Tilbury"
-- }


-- 3. Aggiunte a tabella clients
-- ============================================================
ALTER TABLE public.clients
    ADD COLUMN IF NOT EXISTS consent_social BOOLEAN DEFAULT FALSE;


-- 4. RLS Policies (se abilitato RLS)
-- ============================================================
-- Abilita RLS sulla nuova tabella
ALTER TABLE public.social_content ENABLE ROW LEVEL SECURITY;

-- Policy: ogni tenant vede solo i propri contenuti
CREATE POLICY "tenant_isolation_social_content"
    ON public.social_content
    FOR ALL
    USING (
        tenant_id IN (
            SELECT tenant_id FROM public.memberships
            WHERE user_id = auth.uid()
        )
    );

-- Service role bypassa RLS (usato dal backend FastAPI)
-- Nessuna policy aggiuntiva necessaria per service_role


-- 5. Storage bucket per immagini social
-- ============================================================
-- Esegui via Supabase Dashboard > Storage > New Bucket
-- oppure via API:
-- Nome bucket: social-media
-- Public: true (le immagini devono essere accessibili via URL pubblica)
-- File size limit: 10MB
-- Allowed MIME types: image/jpeg, image/png, image/webp


-- ============================================================
-- VERIFICA MIGRATION
-- ============================================================
-- Esegui dopo per verificare:
-- SELECT column_name, data_type FROM information_schema.columns
-- WHERE table_name = 'social_content' ORDER BY ordinal_position;
