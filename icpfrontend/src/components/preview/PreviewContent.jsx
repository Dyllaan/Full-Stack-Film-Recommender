import useFetchData from '../../hooks/useFetchData';
import { useEffect } from 'react';
import PreviewThumbnail from './PreviewThumbnail';
import Loading from '../Loading';
import { FilmCard } from '../content/FilmCard';
import FilmPosterGrid from './FilmPosterGrid';
import FilmCarousel from './FilmCarousel';

/**
 * Component for showing the thumbnails of preview content on the homepage
 * @author Louis Figes
 */
export function PreviewContent() {
    const { data: films, loading } = useFetchData(`movies`);

    useEffect(() => {
        console.log(films);
    }, [films]);

    if(loading) {
        return <Loading loading={loading} />
    }

    return (
        <FilmCarousel posters={films} />
    );
}

export default PreviewContent;