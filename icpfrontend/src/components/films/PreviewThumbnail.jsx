import PropTypes from 'prop-types';
import RateFilm from './RateFilm';

// Assuming you might want to use icons for the stars, you can import them from a library here

/**
 * Simple component to display an image with a link, used for preview videos
 * @author Louis Figes
 */
function PreviewThumbnail(props) {
  const { title, imgName } = props;
  const moviePoster = `https://image.tmdb.org/t/p/w500/${imgName}`;

  return (
    <div className="relative rounded-lg hover:cursor-pointer overflow-hidden">
      <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-50 flex justify-center items-center gap-2 opacity-0 hover:opacity-100 transition-opacity duration-300">
        <div className="text-yellow-400 flex flex-col max-w-full overflow-hidden">
          <h3 className="text-white truncate z-10 relative p-2">{title}</h3>
          <RateFilm />
        </div>
      </div>
      <img src={moviePoster} alt={title} className="w-full rounded-lg" />
    </div>
  );
}

PreviewThumbnail.propTypes = {
  title: PropTypes.string.isRequired,
  imgName: PropTypes.string.isRequired,
}

export default PreviewThumbnail;
