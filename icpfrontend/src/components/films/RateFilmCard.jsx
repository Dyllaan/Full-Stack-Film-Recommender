import PropTypes from 'prop-types';
import RateFilm from './RateFilm';
import useRatings from '../../hooks/useRatings';

// Assuming you might want to use icons for the stars, you can import them from a library here

/**
 * Simple component to display an image with a link, used for preview videos
 * @author Louis Figes
 */
function RateFilmCard(props) {
  const { film, addRatedFilm } = props;
  const { addRating } = useRatings();

  const moviePoster = `https://image.tmdb.org/t/p/w500/${film.poster_path}`;

  const handleRatingSubmission = (rating, film) => {
    if(addRating(rating, film.movie_slug)) {
      if(addRatedFilm) addRatedFilm(film);
    }
  }

  return (
    <div className="relative rounded-lg hover:cursor-pointer overflow-hidden">
      <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-50 flex justify-center items-center gap-2 opacity-0 hover:opacity-100 transition-opacity duration-300">
        <div className="text-yellow-400 flex flex-col max-w-full overflow-hidden">
          <h3 className="text-white truncate z-10 relative p-2">{film.movie_title}</h3>
          <RateFilm film={film} handleRatingSubmission={handleRatingSubmission} />
        </div>
      </div>
      <img src={moviePoster} alt={film.movie_title} className="w-full rounded-lg" />
    </div>
  );
}

RateFilmCard.propTypes = {
  film: PropTypes.object.isRequired,
  addRatedFilm: PropTypes.func,
}

export default RateFilmCard;
