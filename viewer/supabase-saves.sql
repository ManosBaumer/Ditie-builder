-- Run once in the Supabase SQL Editor.
create table if not exists public.saves (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  city text not null,
  maps_tab text not null default 'all',
  updated_at timestamptz not null default now(),
  unique (user_id, city)
);

alter table public.saves enable row level security;

create policy "saves_select_own"
  on public.saves for select
  to authenticated
  using (auth.uid() = user_id);

create policy "saves_insert_own"
  on public.saves for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "saves_update_own"
  on public.saves for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "saves_delete_own"
  on public.saves for delete
  to authenticated
  using (auth.uid() = user_id);
