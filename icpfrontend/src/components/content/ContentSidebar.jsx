import PropTypes from 'prop-types';
/**
 * 
 * Content sidebar to pick a content item
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

const Sidebar = ({ onSelectItem, content, error, selectedId }) => {

    return (
        <div className="">
            {!error && content.map(item => (
                <div key={item.id} 
                     className={`${selectedId === item.id && 'secondary'} content-item`}
                     onClick={() => onSelectItem(item)}>
                    <p className="truncate text-lg">{item.title}</p>
                    <div className="flex gap-2 text-center">
                        <p className="text-red-300 font-bold">{item.type}</p>
                        {item.award && <p className="award">{item.award} awards</p>}
                        {item.preview_video && <p className="text-red-400 font-bold">Video Preview</p>}
                    </div>
                </div>
            ))}
            {error && <div className="text-red-400">{error}</div>}
        </div>
    );
};

Sidebar.propTypes = {
    onSelectItem: PropTypes.func.isRequired,
    content: PropTypes.array.isRequired,
    error: PropTypes.string,
    selectedId: PropTypes.number,
};

export default Sidebar;
