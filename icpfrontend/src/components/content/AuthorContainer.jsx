import PropTypes from 'prop-types';
/**
 * 
 * Contains all the authors of a piece of content
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */
const AuthorContainer = (props ) => {
  
  const { authorData } = props;

  return (
    <div>
      <p>Authors</p>
      <div className="p-2 flex flex-col lg:flex-row gap-2 w-xs break-words m-2 rounded-lg">
        {authorData.map((item, index) => (
            <div className="text-black p-2 rounded-lg" key={index}>
              <p className="text-neutral-400">{item.author}</p>
              <p className="font-bold text-slate-500">{item.institution}</p>
              <p className="text-neutral-300">{item.city}</p>
              <p className="font-bold text-red-200">{item.country}</p>
            </div>
          ))}
      </div>
    </div>
  );
};

AuthorContainer.propTypes = {
  authorData: PropTypes.array.isRequired,
};

export default AuthorContainer;
