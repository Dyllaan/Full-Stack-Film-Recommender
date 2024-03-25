import { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

/**
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * A button to add or remove a favourite, used on content
 */

function FavouriteButton(props) {
  const { addFavourite, deleteFavourite, contentId, isFavourited } = props;
  const [favourited, setFavourited] = useState(false);
  const image1 = '/heart-red.svg';
  const image2 = '/heart-blue.svg';

  useEffect(() => {
    const checkFavourited = async () => {
      const favouritedStatus = await isFavourited(contentId);
      setFavourited(favouritedStatus);
    };

    checkFavourited();
  }, [contentId, isFavourited]);

  const switchImage = () => {
    if (favourited) {
      handleDeleteFavourite();
    } else {
      handleAddFavourite();
    }
  };

  async function handleDeleteFavourite() {
    await deleteFavourite(contentId);
    setFavourited(false);
  }

  async function handleAddFavourite() {
    await addFavourite(contentId);
    setFavourited(true);
  }

  return (
    <div className="flex justify-center items-center p-1">
      <img
        src={favourited ? image1 : image2}
        alt="Favourite Button"
        className="h-8 hover-scale hover:scale-110 cursor-pointer"
        onClick={switchImage}
      />
    </div>
  );
}

FavouriteButton.propTypes = {
  addFavourite: PropTypes.func.isRequired,
  deleteFavourite: PropTypes.func.isRequired,
  contentId: PropTypes.number.isRequired,
  isFavourited: PropTypes.func.isRequired
};

export default FavouriteButton;
