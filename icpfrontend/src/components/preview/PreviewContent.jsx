import useFetchData from '../../hooks/useFetchData';
import { useEffect } from 'react';
import Loading from '../Loading';
import FilmCarousel from './FilmCarousel';
import { Link } from 'react-router-dom';

/**
 * Component for showing the thumbnails of preview content on the homepage
 * @author Louis Figes
 */
export function PreviewContent() {
    const { data: films, loading } = useFetchData(`recommendations`);

    useEffect(() => {
        console.log(films);
    }, [films]);

    if(loading) {
        return <Loading loading={loading} />
    }

    return (
        <div className="flex flex-col items-center justify-center">
            <h2 className="text-4xl font-bold mt-8">Preview Content</h2>
            {films.length > 0 ? ( <FilmCarousel films={films} /> ) : ( <Link to="/get-started"><p>Lets get you started</p></Link> )}
        </div>
    );
}

export default PreviewContent;