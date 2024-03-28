import PropTypes from 'prop-types';

// Assuming you might want to use icons for the stars, you can import them from a library here

/**
 * Simple component to display an image with a link, used for preview videos
 * @author Louis Figes
 */
function FilmThumbnailCard(props) {
  const { film } = props;
  const moviePoster = `https://image.tmdb.org/t/p/w500/${film.poster_path}`;

  return (
    <div className="w-full">
      <img src={moviePoster} alt={film.movie_title} className="w-full rounded-lg" />
    </div>
  );
}

FilmThumbnailCard.propTypes = {
  film: PropTypes.object.isRequired,
}

export default FilmThumbnailCard;
