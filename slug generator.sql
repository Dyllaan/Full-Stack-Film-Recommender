CREATE OR REPLACE FUNCTION generate_unique_slug() RETURNS VOID AS $$
DECLARE
    rec RECORD;
    base_slug TEXT;
    final_slug TEXT;
    counter INTEGER;
BEGIN
    FOR rec IN SELECT movie_id, movie_title FROM movies LOOP
        base_slug := LOWER(REGEXP_REPLACE(rec.movie_title, '[^a-zA-Z0-9]+', '-', 'g'));
        final_slug := base_slug;
        counter := 1;
        
        LOOP -- Keep trying until a unique slug is found
            IF EXISTS (SELECT 1 FROM movie_slugs WHERE movie_slug = final_slug) THEN
                final_slug := base_slug || '-' || counter::TEXT;
                counter := counter + 1;
            ELSE
                EXIT; -- Exit the loop if the slug is unique
            END IF;
        END LOOP;
        
        -- Attempt to insert the unique slug into movie_slugs
        INSERT INTO movie_slugs (movie_id, movie_slug)
        VALUES (rec.movie_id, final_slug)
        ON CONFLICT (movie_id) 
        DO NOTHING; -- In case another process has already inserted a slug for this movie
    END LOOP;
END;
$$ LANGUAGE plpgsql;
