import useFetchData from '../../hooks/useFetchData';
import LoadingInPage from '../LoadingInPage';
import FilmCarousel from './FilmCarousel';
import fixMovieLensTitle from '../../util/FixMovieLensTitle';

/**
 * Component for showing the thumbnails of preview content on the homepage
 * @author Louis Figes
 */
export function PreviewFilms() {
    const { data: films, loading } = useFetchData(`recommendations`);
    const { data: similarToX, loading: similarLoading } = useFetchData(`similar_movies`);
    const { data: popularFilms, loading: popularLoading } = useFetchData(`best_films`);

    if(loading || similarLoading || popularLoading) {
        return <LoadingInPage loading={loading} />
    }

    const fixedTitle = fixMovieLensTitle(similarToX.similar_to.movie_title);

    return (
        <div className="flex flex-col gap-4 items-center justify-center text-left max-w-[75vw] mx-auto">
            <div>
                {films.length > 0 &&  <FilmCarousel films={films} loading={loading} title="Picks for you"/>}
            </div>
            <div>
                {similarToX.recommendations.length > 0 && <FilmCarousel 
                films={similarToX.recommendations} 
                loading={similarLoading} 
                title={`Because you liked: ${fixedTitle}`}/>}
            </div>
            <div>
                {popularFilms.length > 0 && <FilmCarousel 
                films={popularFilms} 
                loading={popularLoading} 
                title={`Well Rated`}/>}
            </div>
        </div>
    );
}

export default PreviewFilms;