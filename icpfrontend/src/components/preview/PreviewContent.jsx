import useFetchData from '../../hooks/useFetchData';
import Loading from '../Loading';
import FilmCarousel from './FilmCarousel';
import { Link } from 'react-router-dom';

/**
 * Component for showing the thumbnails of preview content on the homepage
 * @author Louis Figes
 */
export function PreviewContent() {
    const { data: films, loading } = useFetchData(`recommendations`);

    if(loading) {
        return <Loading loading={loading} />
    }

    return (
        <div className="flex flex-col items-center justify-center">
            {films && films.length > 0 ? (
                <FilmCarousel films={films} />
            ) : (
                <div className="flex flex-col items-center justify-center">
                    <h3 className="text-2xl font-bold mt-8">No films found</h3>
                    <Link to="/films" className="text-blue-500 underline">View all films</Link>
                </div>
            )}
        </div>
    );
}

export default PreviewContent;