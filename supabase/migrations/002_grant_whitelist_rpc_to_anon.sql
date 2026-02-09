-- Fix 401 on get_whitelist_status: ensure anon can call the RPC.
-- Run this in Supabase Dashboard → SQL Editor if anon gets 401 on signup whitelist check.

-- Ensure function exists (idempotent)
CREATE OR REPLACE FUNCTION public.get_whitelist_status(check_email TEXT)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  row RECORD;
BEGIN
  SELECT role, status INTO row
  FROM public.whitelisted_users
  WHERE email = lower(trim(check_email))
  LIMIT 1;
  IF row IS NULL OR row.status = 'revoked' THEN
    RETURN jsonb_build_object('whitelisted', false, 'role', null);
  END IF;
  RETURN jsonb_build_object('whitelisted', true, 'role', row.role);
END;
$$;

-- Required: allow anonymous (unauthenticated) users to call this for signup
GRANT EXECUTE ON FUNCTION public.get_whitelist_status(TEXT) TO anon;

-- Optional: verify (run separately if you want to test)
-- SELECT public.get_whitelist_status('test@gmail.com');
