import PropTypes from 'prop-types';
import SwipeableTemporaryDrawer from '../Drawer';

/**
 * Simple component to display an image with a link, used for preview videos
 * @author Louis Figes
 */

function PreviewThumbnail(props) {
  const { title, imgName } = props;
  const moviePoster = `https://image.tmdb.org/t/p/w500/${imgName}`;

  return (
    <div className="rounded-lg hover:cursor-pointer hover-scale">
      <p className="text-white truncate">{title}</p>
      <img src={moviePoster} className="w-full rounded-lg" />
    </div>
  );
}

PreviewThumbnail.propTypes = {
  title: PropTypes.string.isRequired,
  imgName: PropTypes.string.isRequired,
}

export default PreviewThumbnail;